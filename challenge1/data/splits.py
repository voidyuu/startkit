from __future__ import annotations

from collections.abc import Iterable

from braindecode.datasets import BaseConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

REMOVED_EVAL_SUBJECTS = {
    "NDARWV769JM7",
    "NDARME789TD2",
    "NDARUA442ZVF",
    "NDARJP304NK1",
    "NDARTY128YLU",
    "NDARDW550GU6",
    "NDARLD243KRE",
    "NDARUJ292JXV",
    "NDARBA381JGH",
}


def release_name(release: int | str) -> str:
    release_str = str(release)
    return release_str if release_str.startswith("R") else f"R{release_str}"


def split_window_dataset_by_subject(windows, train_subjects, valid_subjects, test_subjects):
    return _split_dataset_by_subject(windows, train_subjects, valid_subjects, test_subjects)


def split_recording_dataset_by_subject(recordings, train_subjects, valid_subjects, test_subjects):
    return _split_dataset_by_subject(recordings, train_subjects, valid_subjects, test_subjects)


def get_dataset_subjects(dataset) -> list[str]:
    subjects = []

    for recording in dataset.datasets:
        description = getattr(recording, "description", None)
        if description is None or "subject" not in description:
            raise RuntimeError("Dataset recording is missing 'subject' in its description.")
        subjects.append(str(description["subject"]))

    return subjects


def split_eval_subjects(meta_information, *, random_seed: int, valid_frac: float = 0.5):
    return split_eval_subjects_from_subjects(
        meta_information["subject"].unique(),
        random_seed=random_seed,
        valid_frac=valid_frac,
    )


def split_eval_subjects_from_subjects(
    subjects: Iterable[str],
    *,
    random_seed: int,
    valid_frac: float = 0.5,
):
    subjects = _get_eval_subjects_from_subjects(subjects)

    valid_subjects, test_subjects = train_test_split(
        subjects,
        test_size=(1 - valid_frac),
        random_state=check_random_state(random_seed),
        shuffle=True,
    )
    return set(valid_subjects), set(test_subjects)


def split_train_valid_test_subjects(
    meta_information,
    *,
    random_seed: int,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
):
    return split_train_valid_test_subjects_from_subjects(
        meta_information["subject"].unique(),
        random_seed=random_seed,
        train_frac=train_frac,
        valid_frac=valid_frac,
    )


def split_train_valid_test_subjects_from_subjects(
    subjects: Iterable[str],
    *,
    random_seed: int,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
):
    subjects = _get_eval_subjects_from_subjects(subjects)
    if len(subjects) < 3:
        raise RuntimeError("Need at least 3 subjects to split one release into train/valid/test sets.")
    if train_frac <= 0 or valid_frac <= 0 or train_frac + valid_frac >= 1:
        raise ValueError("Expected train_frac > 0, valid_frac > 0, and train_frac + valid_frac < 1.")

    shuffled_subjects = list(subjects)
    rng = check_random_state(random_seed)
    rng.shuffle(shuffled_subjects)

    n_subjects = len(shuffled_subjects)
    n_train = max(1, int(n_subjects * train_frac))
    n_valid = max(1, int(n_subjects * valid_frac))

    if n_train >= n_subjects - 1:
        n_train = n_subjects - 2
    if n_train < 1:
        n_train = 1

    max_valid = n_subjects - n_train - 1
    if max_valid < 1:
        raise RuntimeError("Could not reserve at least one subject for each same-release split.")
    n_valid = min(n_valid, max_valid)

    train_subjects = set(shuffled_subjects[:n_train])
    valid_subjects = set(shuffled_subjects[n_train : n_train + n_valid])
    test_subjects = set(shuffled_subjects[n_train + n_valid :])
    if not valid_subjects or not test_subjects:
        raise RuntimeError("Same-release split produced an empty validation or test subject set.")

    return train_subjects, valid_subjects, test_subjects


def _split_dataset_by_subject(dataset, train_subjects, valid_subjects, test_subjects):
    subject_split = dataset.split("subject")
    train_set, valid_set, test_set = [], [], []

    for subject in subject_split:
        if subject in train_subjects:
            train_set.append(subject_split[subject])
        elif subject in valid_subjects:
            valid_set.append(subject_split[subject])
        elif subject in test_subjects:
            test_set.append(subject_split[subject])

    def concat_or_none(datasets):
        return BaseConcatDataset(datasets) if datasets else None

    return (
        concat_or_none(train_set),
        concat_or_none(valid_set),
        concat_or_none(test_set),
    )


def _get_eval_subjects(meta_information) -> list[str]:
    return _get_eval_subjects_from_subjects(meta_information["subject"].unique())


def _get_eval_subjects_from_subjects(subjects: Iterable[str]) -> list[str]:
    unique_subjects = list(dict.fromkeys(str(subject) for subject in subjects))
    return [subject for subject in unique_subjects if subject not in REMOVED_EVAL_SUBJECTS]
