from src.structures import PatientBST

def test_patient_bst_insert_and_search():
    records = [
        {"age": 30, "name": "p1"},
        {"age": 20, "name": "p2"},
        {"age": 40, "name": "p3"},
    ]

    bst = PatientBST(key_fn=lambda r: r["age"])
    for r in records:
        bst.insert(r)

    assert len(bst) == 3
    found = bst.search(20.0)
    assert found is not None
    assert found["name"] == "p2"

def test_patient_bst_inorder_sorted():
    records = [{"age": 50}, {"age": 10}, {"age": 30}]
    bst = PatientBST(key_fn=lambda r: r["age"])
    for r in records:
        bst.insert(r)

    inorder = bst.inorder()
    ages = [r["age"] for r in inorder]
    assert ages == sorted(ages)
