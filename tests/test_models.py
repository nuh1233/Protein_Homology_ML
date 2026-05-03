from __future__ import annotations

import numpy as np
from scipy import sparse

from cafa6.models import predict_branch_scores, predict_branch_topk, train_branch_model


def test_train_branch_model_and_predict_topk() -> None:
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype="float32",
    )
    y = sparse.csr_matrix(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )

    model = train_branch_model(
        x,
        y,
        label_terms=["GO:1", "GO:2"],
        branch="MF",
        max_iter=5,
        random_state=1,
        n_jobs=1,
    )
    scores = predict_branch_scores(model, x)
    predictions = predict_branch_topk(model, x, ["P1", "P2", "P3", "P4"], top_k=1)

    assert scores.shape == (4, 2)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
    assert set(predictions.columns) == {"entry_id", "term", "branch", "score"}
    assert len(predictions) == 4
