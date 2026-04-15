from __future__ import annotations

import json
import os
import random
import socket
import ssl
import urllib.parse
import urllib.request
from contextlib import contextmanager

import socks

_ORIGINAL_SOCKET = socket.socket
_ORIGINAL_CREATE_CONNECTION = socket.create_connection
_ORIGINAL_GETADDRINFO = socket.getaddrinfo
_PATCHED = False
_ORIGINAL_PYMONGO_SYNC_GET_HOSTS = None
_ORIGINAL_PYMONGO_SYNC_GET_HOSTS_AND_MIN_TTL = None
_ORIGINAL_PYMONGO_SYNC_GET_OPTIONS = None

_PATCH_STATE: dict[str, object] = {
    "enabled": False,
    "doh_url": None,
    "proxy_url": None,
    "host_overrides": {},
    "pymongo_doh_fallback": False,
}


@contextmanager
def _use_original_network_stack():
    saved_socket = socket.socket
    saved_create_connection = socket.create_connection
    saved_getaddrinfo = socket.getaddrinfo
    socket.socket = _ORIGINAL_SOCKET
    socket.create_connection = _ORIGINAL_CREATE_CONNECTION
    socket.getaddrinfo = _ORIGINAL_GETADDRINFO
    try:
        yield
    finally:
        socket.socket = saved_socket
        socket.create_connection = saved_create_connection
        socket.getaddrinfo = saved_getaddrinfo


def _parse_host_overrides(raw_value: str | None) -> dict[str, tuple[str, ...]]:
    if not raw_value:
        return {}
    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError("STARTKIT_DNS_HOSTS must be a JSON object mapping hostnames to IPs.")

    overrides: dict[str, tuple[str, ...]] = {}
    for host, value in parsed.items():
        if isinstance(value, str):
            values = (value,)
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            values = tuple(value)
        else:
            raise ValueError("STARTKIT_DNS_HOSTS values must be strings or arrays of strings.")
        overrides[str(host).lower()] = values
    return overrides


def _normalize_host(host: str | bytes | None) -> str | None:
    if host is None:
        return None
    if isinstance(host, bytes):
        host = host.decode("idna")
    if host.endswith("."):
        host = host[:-1]
    return host.lower()


def _is_ip_address(host: str | None) -> bool:
    if not host:
        return False
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            socket.inet_pton(family, host)
            return True
        except OSError:
            continue
    return False


def _build_sockaddr(family: int, ip_address: str, port: int | str) -> tuple:
    if family == socket.AF_INET6:
        return (ip_address, port, 0, 0)
    return (ip_address, port)


def _normalize_socktype(socktype: int) -> int:
    if socktype == 0:
        return socket.SOCK_STREAM

    normalized = socktype
    for extra_flag in (
        getattr(socket, "SOCK_CLOEXEC", 0),
        getattr(socket, "SOCK_NONBLOCK", 0),
    ):
        if extra_flag:
            normalized &= ~int(extra_flag)

    if normalized in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
        return normalized
    return socket.SOCK_STREAM


def _decode_txt_record(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value.replace('\\"', '"')


def _doh_query(name: str, record_type: str, timeout: float, doh_url: str) -> list[dict]:
    query = urllib.parse.urlencode({"name": name, "type": record_type})
    ssl_context = ssl._create_unverified_context()
    request = urllib.request.Request(
        f"{doh_url}?{query}",
        headers={
            "accept": "application/dns-json",
            "user-agent": "startkit-network/1.0",
        },
    )
    with _use_original_network_stack():
        with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
            payload = json.load(response)

    if payload.get("Status") != 0:
        return []
    return payload.get("Answer", [])


def _resolve_host_via_doh(
    host: str,
    family: int,
    timeout: float,
    doh_url: str,
) -> tuple[tuple[int, str], ...]:
    families = (
        (socket.AF_INET, "A", 1),
        (socket.AF_INET6, "AAAA", 28),
    )
    if family == socket.AF_INET:
        families = ((socket.AF_INET, "A", 1),)
    elif family == socket.AF_INET6:
        families = ((socket.AF_INET6, "AAAA", 28),)

    resolved: list[tuple[int, str]] = []
    for current_family, record_type, record_code in families:
        answers = _doh_query(host, record_type, timeout, doh_url)
        aliases = {host.rstrip(".").lower()}
        for answer in answers:
            answer_type = int(answer.get("type", -1))
            answer_name = str(answer.get("name", "")).rstrip(".").lower()
            data = str(answer.get("data", "")).strip().rstrip(".")
            if answer_type == 5 and answer_name in aliases and data:
                aliases.add(data.lower())
                continue
            if answer_type != record_code:
                continue
            if data and answer_name in aliases:
                resolved.append((current_family, data))

    if not resolved:
        raise socket.gaierror(f"DoH lookup returned no A/AAAA records for {host!r}.")
    return tuple(resolved)


def _pymongo_srv_hosts_from_doh(
    fqdn: str,
    srv_service_name: str,
    timeout: float,
    srv_max_hosts: int,
    plist: list[str],
    slen: int,
    nparts: int,
    doh_url: str,
) -> tuple[list[tuple[str, int]], int]:
    answers = _doh_query(f"_{srv_service_name}._tcp.{fqdn}", "SRV", timeout, doh_url)
    nodes: list[tuple[str, int]] = []
    ttl_values: list[int] = []

    for answer in answers:
        data = str(answer.get("data", "")).strip()
        parts = data.split()
        if len(parts) != 4:
            continue
        _, _, port_text, host_text = parts
        host = host_text.rstrip(".")
        port = int(port_text)
        ttl_values.append(int(answer.get("TTL", 0)))
        nodes.append((host, port))

    if not nodes:
        raise socket.gaierror(f"DoH lookup returned no SRV records for {fqdn!r}.")

    for host, _ in nodes:
        srv_host = host.lower()
        if fqdn == srv_host and nparts < 3:
            raise ValueError("Invalid SRV host: return address is identical to SRV hostname")
        nlist = srv_host.split(".")[1:][-slen:]
        if plist != nlist:
            raise ValueError(f"Invalid SRV host: {host}")

    if srv_max_hosts:
        nodes = random.sample(nodes, min(srv_max_hosts, len(nodes)))

    min_ttl = min(ttl_values) if ttl_values else 0
    return nodes, min_ttl


def _pymongo_txt_options_from_doh(fqdn: str, timeout: float, doh_url: str) -> str | None:
    answers = _doh_query(fqdn, "TXT", timeout, doh_url)
    txt_values = [_decode_txt_record(str(answer.get("data", ""))) for answer in answers]
    txt_values = [value for value in txt_values if value]
    if not txt_values:
        return None
    if len(txt_values) > 1:
        raise ValueError("Only one TXT record is supported")
    return txt_values[0]


def _patch_pymongo_srv_resolver_with_doh(doh_url: str) -> bool:
    global _ORIGINAL_PYMONGO_SYNC_GET_HOSTS
    global _ORIGINAL_PYMONGO_SYNC_GET_HOSTS_AND_MIN_TTL
    global _ORIGINAL_PYMONGO_SYNC_GET_OPTIONS

    try:
        from pymongo.synchronous import srv_resolver as sync_srv_resolver
    except Exception:
        return False

    if _ORIGINAL_PYMONGO_SYNC_GET_HOSTS is not None:
        return False

    _ORIGINAL_PYMONGO_SYNC_GET_HOSTS = sync_srv_resolver._SrvResolver.get_hosts
    _ORIGINAL_PYMONGO_SYNC_GET_HOSTS_AND_MIN_TTL = sync_srv_resolver._SrvResolver.get_hosts_and_min_ttl
    _ORIGINAL_PYMONGO_SYNC_GET_OPTIONS = sync_srv_resolver._SrvResolver.get_options

    def get_hosts(self):
        try:
            nodes, _ = _pymongo_srv_hosts_from_doh(
                fqdn=self._SrvResolver__fqdn,
                srv_service_name=self._SrvResolver__srv,
                timeout=self._SrvResolver__connect_timeout,
                srv_max_hosts=self._SrvResolver__srv_max_hosts,
                plist=self._SrvResolver__plist,
                slen=self._SrvResolver__slen,
                nparts=self.nparts,
                doh_url=doh_url,
            )
            return nodes
        except Exception:
            return _ORIGINAL_PYMONGO_SYNC_GET_HOSTS(self)

    def get_hosts_and_min_ttl(self):
        try:
            return _pymongo_srv_hosts_from_doh(
                fqdn=self._SrvResolver__fqdn,
                srv_service_name=self._SrvResolver__srv,
                timeout=self._SrvResolver__connect_timeout,
                srv_max_hosts=self._SrvResolver__srv_max_hosts,
                plist=self._SrvResolver__plist,
                slen=self._SrvResolver__slen,
                nparts=self.nparts,
                doh_url=doh_url,
            )
        except Exception:
            return _ORIGINAL_PYMONGO_SYNC_GET_HOSTS_AND_MIN_TTL(self)

    def get_options(self):
        try:
            return _pymongo_txt_options_from_doh(
                fqdn=self._SrvResolver__fqdn,
                timeout=self._SrvResolver__connect_timeout,
                doh_url=doh_url,
            )
        except Exception:
            return _ORIGINAL_PYMONGO_SYNC_GET_OPTIONS(self)

    sync_srv_resolver._SrvResolver.get_hosts = get_hosts
    sync_srv_resolver._SrvResolver.get_hosts_and_min_ttl = get_hosts_and_min_ttl
    sync_srv_resolver._SrvResolver.get_options = get_options
    return True


def install_network(
    *,
    doh_url: str,
    proxy_url: str | None = None,
    host_overrides: dict[str, tuple[str, ...]] | None = None,
    timeout: float = 5.0,
) -> bool:
    global _PATCHED
    if _PATCHED:
        return False

    overrides = {key.lower(): tuple(value) for key, value in (host_overrides or {}).items()}

    if proxy_url:
        parsed = urllib.parse.urlparse(proxy_url)
        if parsed.scheme.lower() != "socks5":
            raise ValueError(f"Unsupported proxy scheme: {parsed.scheme!r}")
        if not parsed.hostname or not parsed.port:
            raise ValueError("SOCKS5 proxy URL must include host and port.")
        socks.set_default_proxy(
            socks.SOCKS5,
            addr=parsed.hostname,
            port=parsed.port,
            username=parsed.username,
            password=parsed.password,
            rdns=True,
        )

        class StartkitSocksSocket(socks.socksocket):
            def __init__(self, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, *args, **kwargs):
                super().__init__(family, _normalize_socktype(type), proto, *args, **kwargs)

        socket.socket = StartkitSocksSocket

        def proxied_create_connection(address, timeout=None, source_address=None):
            host, port = address
            sock = StartkitSocksSocket()
            if timeout is not None:
                sock.settimeout(timeout)
            if source_address is not None:
                sock.bind(source_address)
            sock.connect((host, port))
            return sock

        socket.create_connection = proxied_create_connection
    else:
        socket.socket = _ORIGINAL_SOCKET
        socket.create_connection = _ORIGINAL_CREATE_CONNECTION

    def custom_getaddrinfo(
        host: str | bytes | None,
        port,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ):
        normalized_host = _normalize_host(host)
        if normalized_host is None or normalized_host in {"", "localhost"} or _is_ip_address(normalized_host):
            return _ORIGINAL_GETADDRINFO(host, port, family, type, proto, flags)

        normalized_type = _normalize_socktype(type)

        override_values = overrides.get(normalized_host)
        if override_values:
            resolved = []
            for value in override_values:
                current_family = socket.AF_INET6 if ":" in value else socket.AF_INET
                if family in (socket.AF_UNSPEC, current_family):
                    resolved.append(
                        (
                            current_family,
                            normalized_type,
                            proto,
                            "",
                            _build_sockaddr(current_family, value, port),
                        )
                    )
            if resolved:
                return resolved
            raise socket.gaierror(f"No custom DNS records match requested family for {normalized_host!r}.")

        if proxy_url:
            family_out = socket.AF_INET6 if family == socket.AF_INET6 else socket.AF_INET
            return [(family_out, normalized_type, proto, "", (normalized_host, port))]

        resolved_hosts = _resolve_host_via_doh(normalized_host, family, timeout, doh_url)
        return [
            (resolved_family, normalized_type, proto, "", _build_sockaddr(resolved_family, ip_address, port))
            for resolved_family, ip_address in resolved_hosts
        ]

    socket.getaddrinfo = custom_getaddrinfo
    pymongo_doh_fallback = _patch_pymongo_srv_resolver_with_doh(doh_url)
    _PATCHED = True
    _PATCH_STATE["enabled"] = True
    _PATCH_STATE["doh_url"] = doh_url
    _PATCH_STATE["proxy_url"] = proxy_url
    _PATCH_STATE["host_overrides"] = overrides
    _PATCH_STATE["pymongo_doh_fallback"] = pymongo_doh_fallback
    if proxy_url:
        parsed = urllib.parse.urlparse(proxy_url)
        print(f"Custom network enabled via DoH={doh_url} and SOCKS5={parsed.hostname}:{parsed.port}")
    else:
        print(f"Custom network enabled via DoH={doh_url}")
    return True


def install_network_from_env() -> bool:
    doh_url = os.environ.get("STARTKIT_DOH_URL", "https://223.5.5.5/resolve")
    proxy_url = (
        os.environ.get("STARTKIT_SOCKS_PROXY")
        or os.environ.get("all_proxy")
        or os.environ.get("ALL_PROXY")
    )
    timeout = float(os.environ.get("STARTKIT_DNS_TIMEOUT", "5"))
    host_overrides = _parse_host_overrides(os.environ.get("STARTKIT_DNS_HOSTS"))
    return install_network(
        doh_url=doh_url,
        proxy_url=proxy_url,
        host_overrides=host_overrides,
        timeout=timeout,
    )


def network_state() -> dict[str, object]:
    return dict(_PATCH_STATE)
