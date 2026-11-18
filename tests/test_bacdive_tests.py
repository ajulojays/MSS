"""BacDive test extraction (utilization, enzymes, production, halophily) and trait labels."""
import pandas as pd

from mss.bacdive_tests import parse_tests, trait_labels


def _rec(util="+", enz="+", prod="yes"):
    return {
        "Physiology and metabolism": {
            "metabolite utilization": [{"@ref": 1, "Chebi-ID": 17234, "metabolite": "glucose",
                                        "utilization activity": util, "kind of utilization tested": "carbon source"}],
            "enzymes": [{"@ref": 2, "value": "catalase", "activity": enz, "ec": "1.11.1.6"}],
            "metabolite production": [{"@ref": 3, "Chebi-ID": 35581, "metabolite": "indole", "production": prod}],
            "halophily": [{"@ref": 4, "salt": "NaCl", "growth": "positive", "tested relation": "growth",
                           "concentration": "0-2 %"}],
        },
        "Isolation, sampling and environmental information": {
            "isolation source categories": [{"Cat1": "#Host"}]},
    }


def test_parse_tests_keeps_bacdive_wording():
    rows = pd.DataFrame(parse_tests(7, _rec()))
    assert set(rows.section) == {"metabolite_utilization", "enzymes", "metabolite_production", "halophily"}
    u = rows[rows.section == "metabolite_utilization"].iloc[0]
    assert (u["item"], u["item_id"], u["detail"], u["result"]) == ("glucose", "17234", "carbon source", "+")
    assert rows[rows.section == "halophily"].iloc[0].detail == "growth; 0-2 %"
    # API envelope is unwrapped the same way
    assert len(parse_tests(7, {"results": {"7": _rec()}})) == len(rows)


def test_trait_labels_min_class_conflicts_and_drops():
    recs = {i: _rec("+" if i < 6 else "-", "+" if i % 2 else "-") for i in range(10)}
    recs[10] = _rec("+/-")                                    # rare class, left out
    t = pd.DataFrame([r for i, rec in recs.items() for r in parse_tests(i, rec)])
    # strain 11: disagreeing entries for the same test -> conflict, no label
    t = pd.concat([t, pd.DataFrame(parse_tests(11, _rec("+"))), pd.DataFrame(parse_tests(11, _rec("-")))])
    lab, cat = trait_labels(t, range(12), min_class=3)
    col = "util__chebi17234__carbon_source"
    assert col in lab.columns and "enz__catalase" in lab.columns
    L = lab.set_index("bacdive_id")
    assert L.loc[10, col] == "" and "+/-=1" in cat.set_index("trait").loc[col, "classes_left_out"]
    assert L.loc[11, col] == "" and bool(L.loc[11, f"{col}_conflict"])
    assert cat.set_index("trait").loc["prod__chebi35581", "status"].startswith("dropped")  # one class only
