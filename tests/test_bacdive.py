"""BacDive parsing tests on hand-written fixtures shaped like API v2 records (no network)."""
import pandas as pd
import pytest

from mss.bacdive import BacDiveClient, build_reference, parse_record, summarize_reference
from mss.genomes import make_manifest
from mss.ontology import load_oxygen_map, normalize_species


def rec(oxygen=None, morph=None, spores=None, temps=None, genomes=None, gbp=None, species="Testus alpha",
        ph=None, salt=None):
    phys = {}
    if salt is not None:
        phys["halophily"] = salt
    if oxygen is not None:
        phys["oxygen tolerance"] = oxygen
    if spores is not None:
        phys["spore formation"] = spores
    r = {
        "General": {"doi": "10.0/test"},
        "Name and taxonomic classification": {"species": species, "genus": species.split()[0],
                                              "family": "Testaceae", "type strain": "yes"},
        "Physiology and metabolism": phys,
        "Culture and growth conditions": {"culture temp": temps or [], "culture pH": ph or []},
        "Morphology": {"cell morphology": morph} if morph is not None else {},
        "Sequence information": {"Genome sequences": genomes or []},
    }
    if gbp is not None:
        r["Genome-based predictions"] = gbp
    return r


def test_single_dict_and_list_forms():
    a = parse_record(1, rec(oxygen={"@ref": 5, "oxygen tolerance": "anaerobe"}))
    b = parse_record(2, rec(oxygen=[{"@ref": 5, "oxygen tolerance": "anaerobe"},
                                    {"@ref": 6, "oxygen tolerance": "anaerobe"}]))
    assert a["oxygen"] == b["oxygen"] == "anaerobe"
    assert not a["oxygen_conflict"] and not b["oxygen_conflict"]
    assert "oxygen:5,6" in b["evidence_references"]


def test_oxygen_normalization_and_strictness():
    r = parse_record(1, rec(oxygen=[{"@ref": 1, "oxygen tolerance": "obligate anaerobe"},
                                    {"@ref": 2, "oxygen tolerance": "anaerobe"}]))
    assert r["oxygen"] == "anaerobe" and not r["oxygen_conflict"]
    # obligate is kept separately from generic; generic is never upgraded
    assert r["oxygen_detail"] == "anaerobe:obligate; anaerobe:unspecified"
    assert r["oxygen_raw"] == "obligate anaerobe; anaerobe"
    g = parse_record(2, rec(oxygen={"@ref": 1, "oxygen tolerance": "anaerobe"}))
    assert g["oxygen_detail"] == "anaerobe:unspecified"
    f = parse_record(3, rec(oxygen={"@ref": 1, "oxygen tolerance": "facultative anaerobe"}))
    assert f["oxygen"] == "facultative"
    m = parse_record(4, rec(oxygen={"@ref": 1, "oxygen tolerance": "Microaerophile"}))
    assert m["oxygen"] == "microaerophile"


def test_conflict_preserved_not_voted():
    r = parse_record(1, rec(oxygen=[{"@ref": 1, "oxygen tolerance": "anaerobe"},
                                    {"@ref": 2, "oxygen tolerance": "anaerobe"},
                                    {"@ref": 3, "oxygen tolerance": "aerobe"}]))
    assert r["oxygen"] == "" and r["oxygen_conflict"] is True
    assert r["oxygen_raw"] == "anaerobe; anaerobe; aerobe"


def test_owner_mapped_terms():
    fa = parse_record(1, rec(oxygen={"@ref": 1, "oxygen tolerance": "facultative aerobe"}))
    assert fa["oxygen"] == "facultative" and fa["oxygen_raw"] == "facultative aerobe"
    mt = parse_record(2, rec(oxygen={"@ref": 1, "oxygen tolerance": "microaerotolerant"}))
    assert mt["oxygen"] == "microaerophile" and mt["oxygen_raw"] == "microaerotolerant"


def test_unmapped_strings_left_empty_and_overridable(tmp_path):
    r = parse_record(1, rec(oxygen={"@ref": 1, "oxygen tolerance": "oxygen-sensitive thing"}))
    assert r["oxygen"] == "" and r["oxygen_unmapped"] == "oxygen-sensitive thing"
    p = tmp_path / "map.tsv"
    p.write_text("raw\tcategory\noxygen-sensitive thing\tanaerobe\nmicroaerotolerant\tunmapped\n")
    m = load_oxygen_map(p)
    r2 = parse_record(1, rec(oxygen={"@ref": 1, "oxygen tolerance": "oxygen-sensitive thing"}), m)
    assert r2["oxygen"] == "anaerobe"
    r3 = parse_record(1, rec(oxygen={"@ref": 1, "oxygen tolerance": "microaerotolerant"}), m)
    assert r3["oxygen"] == "" and r3["oxygen_unmapped"] == "microaerotolerant"


def test_missing_is_not_negative():
    r = parse_record(1, rec())
    for t in ["oxygen", "motility", "sporulation", "gram_stain"]:
        assert r[t] == ""
        assert r[f"{t}_conflict"] is False


def test_binary_traits_and_conflicts():
    r = parse_record(1, rec(morph=[{"@ref": 1, "gram stain": "negative", "motility": "yes"},
                                   {"@ref": 2, "gram stain": "positive", "motility": "yes"}],
                            spores={"@ref": 3, "spore formation": "no"}))
    assert r["motility"] == "yes"
    assert r["gram_stain"] == "" and r["gram_stain_conflict"]
    assert r["sporulation"] == "no"


def test_genome_based_predictions_ignored():
    r = parse_record(1, rec(gbp={"prediction": [{"trait": "oxygen", "value": "aerobe"}]},
                            morph={"@ref": 1, "motility": "no"}))
    assert r["oxygen"] == ""  # predictions never become reference labels


def test_optimum_temperature():
    r = parse_record(1, rec(temps=[{"@ref": 1, "type": "growth", "temperature": "30", "growth": "positive"},
                                   {"@ref": 2, "type": "optimum", "temperature": "37"}]))
    assert r["optimal_temperature_c"] == 37.0
    r = parse_record(1, rec(temps=[{"@ref": 2, "type": "optimum", "temperature": "30-37"}]))
    assert r["optimal_temperature_c"] is None and r["optimal_temperature_raw"] == "30-37"


def test_genomes_and_manifest():
    g = [{"INSDC accession": "GCA_1", "score": 50}, {"INSDC accession": "GCA_2", "score": 90}]
    recs = [(1, rec(genomes=g, oxygen={"@ref": 1, "oxygen tolerance": "aerobe"})),
            (2, rec(genomes=[{"INSDC accession": "GCA_2", "score": 90}], species="Testus beta")),
            (3, rec(species="Testus gamma"))]
    ref = build_reference(recs)
    assert ref.loc[0, "genome_accession"] == "GCA_2"
    assert ref.loc[0, "all_genome_accessions"] == "GCA_1;GCA_2"
    m, dup = make_manifest(ref)
    assert list(m["accession"]) == ["GCA_2"] and len(dup) == 1
    s = summarize_reference(ref)
    assert s["n_strains"] == 3 and s["n_with_genome"] == 2


def test_species_normalization():
    assert normalize_species("s__Bacteroides_fragilis") == "bacteroides fragilis"
    assert normalize_species("k__Bacteria|g__Bacteroides|s__Bacteroides_fragilis") == "bacteroides fragilis"
    assert normalize_species("Blautia_A obeum") == "blautia_a obeum"
    assert normalize_species("Blautia obeum") != normalize_species("Blautia_A obeum")


class FakeResp:
    def __init__(self, code, payload, headers=None):
        self.status_code, self._p, self.headers, self.text = code, payload, headers or {}, ""

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, responses):
        self.responses, self.urls, self.headers = list(responses), [], {}

    def get(self, url, timeout=None):
        self.urls.append(url)
        return self.responses.pop(0)


def test_client_retry_pagination_and_cache(tmp_path):
    s = FakeSession([
        FakeResp(429, {}, {"Retry-After": "0"}),
        FakeResp(200, {"count": 3, "next": "https://x/page2", "results": [1, 2]}),
        FakeResp(200, {"count": 3, "next": None, "results": [3]}),
        FakeResp(200, {"count": 3, "next": None, "results": {"1": rec(), "2": rec(), "3": rec()}}),
    ])
    c = BacDiveClient(cache_dir=tmp_path, session=s, backoff=0, pause=0)
    ids = c.search_name("Testus alpha")
    assert ids == [1, 2, 3]
    assert s.urls[0].endswith("/v2/taxon/Testus/alpha")
    got = dict(c.fetch(ids))
    assert sorted(got) == [1, 2, 3]
    assert (tmp_path / "2.json").exists()
    # second fetch is served from cache: no more HTTP calls
    n = len(s.urls)
    assert sorted(dict(c.fetch([1, 2]))) == [1, 2] and len(s.urls) == n


def test_client_gives_up(tmp_path):
    s = FakeSession([FakeResp(503, {})] * 3)
    c = BacDiveClient(session=s, backoff=0, pause=0, max_retries=2)
    with pytest.raises(RuntimeError):
        c.search_taxon("Testus")


def test_cell_shape_ph_salt_and_growth_ranges():
    r = parse_record(1, rec(
        morph=[{"@ref": 1, "cell shape": "rod-shaped"}, {"@ref": 2, "cell shape": "Rod-shaped"}],
        temps=[{"@ref": 1, "type": "growth", "growth": "positive", "temperature": "5-30"},
               {"@ref": 1, "type": "growth", "growth": "negative", "temperature": "41"},
               {"@ref": 2, "type": "growth", "growth": "positive", "temperature": "37"}],
        ph=[{"@ref": 3, "ability": "positive", "type": "growth", "pH": "6-9", "PH range": "alkaliphile"},
            {"@ref": 3, "ability": "positive", "type": "optimum", "pH": "8"}],
        salt=[{"@ref": 4, "salt": "NaCl", "growth": "positive", "tested relation": "growth", "concentration": "0.5-10 %"},
              {"@ref": 4, "salt": "NaCl", "growth": "no", "tested relation": "growth", "concentration": "15 %"},
              {"@ref": 4, "salt": "NaCl", "growth": "positive", "tested relation": "optimum", "concentration": "3.5 %"},
              {"@ref": 5, "salt": "NaCl", "growth": "positive", "tested relation": "growth", "concentration": "2 g/L"}]))
    assert r["cell_shape"] == "rod-shaped" and not r["cell_shape_conflict"]
    assert (r["growth_temperature_min_c"], r["growth_temperature_max_c"]) == (5.0, 37.0)  # negative 41 ignored
    assert r["ph_range"] == "alkaliphile" and r["optimal_ph"] == 8.0
    assert (r["growth_ph_min"], r["growth_ph_max"]) == (6.0, 9.0)
    assert r["optimal_nacl_pct"] == 3.5
    assert r["growth_nacl_max_pct"] == 10.0  # 15 % was 'no'; g/L entry not converted
    assert "nacl:4,5" in r["evidence_references"]
    s = parse_record(2, rec(morph=[{"@ref": 1, "cell shape": "rod-shaped"},
                                   {"@ref": 2, "cell shape": "coccus-shaped"}]))
    assert s["cell_shape"] == "" and s["cell_shape_conflict"]
    o = parse_record(3, rec(morph={"@ref": 1, "cell shape": "ovoid-shaped"}))
    assert o["cell_shape"] == "ovoid-shaped"  # BacDive string kept, not merged with "oval-shaped"


def test_contradicted_growth_range_is_flagged():
    r = parse_record(1, rec(salt=[
        {"@ref": 1, "salt": "NaCl", "growth": "positive", "tested relation": "growth", "concentration": "10 %"},
        {"@ref": 2, "salt": "NaCl", "growth": "no", "tested relation": "growth", "concentration": "10 %"}]))
    assert r["growth_nacl_max_pct"] is None and r["growth_nacl_conflict"] is True
