from src.structures import HashEncoder

def test_hash_encoder_fit_transform():
    values = ["A", "B", "A", "C"]
    enc = HashEncoder()
    out = enc.fit_transform(values)

    # mêmes entrées => mêmes codes
    assert out[0] == out[2]
    # taille mapping = nb catégories
    assert len(enc) == 3
    # tous les codes sont des ints >= 0
    assert all(isinstance(x, int) and x >= 0 for x in out)

def test_hash_encoder_unknown_value():
    enc = HashEncoder().fit(["X", "Y"])
    out = enc.transform(["X", "Z"])
    assert out[0] != -1
    assert out[1] == -1
