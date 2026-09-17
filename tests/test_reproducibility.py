"""Reproducibility.

The critique rests in part on the evaluator's construction being difficult to
reproduce. As such, the standard we hold ourselves to here is a strict one.
"""

from model_scores import score


def test_the_same_seed_gives_the_same_answer(reference, fat_tailed_model):
    a = score(reference, fat_tailed_model, margin=0.10, seed=11)
    b = score(reference, fat_tailed_model, margin=0.10, seed=11)
    assert a.to_dict() == b.to_dict()


def test_a_different_seed_gives_a_different_bootstrap(reference, fat_tailed_model):
    a = score(reference, fat_tailed_model, margin=0.10, seed=11)
    c = score(reference, fat_tailed_model, margin=0.10, seed=12)
    assert a.to_dict() != c.to_dict()


def test_a_pair_does_not_depend_on_where_it_fell_in_the_batch(reference, fat_tailed_model):
    """Each pair draws its own stream, so a pair can be rerun on its own and give
    the same answer it gave inside the batch.
    """
    from model_scores import score_all
    keys = {("a",): reference, ("b",): reference, ("c",): reference}
    models = {("a",): fat_tailed_model, ("b",): fat_tailed_model, ("c",): fat_tailed_model}
    results = score_all(keys, models, margin=0.10)
    assert results[0].to_dict()["quantities"] == results[2].to_dict()["quantities"]
