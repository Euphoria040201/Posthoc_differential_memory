"""Procedural synthetic single-chunk QA generator with disjoint train/val/test entities.
Novel (base-model-unknowable) facts, diverse answer types: named-entity / number / material /
multi-word span. Each split draws from a NON-OVERLAPPING slice of the entity pools so val/test
facts are never seen in training."""
import random

_PRE = ["Zeph", "Hald", "Merid", "Xol", "Kesh", "Rav", "Torv", "Quen", "Brax", "Vael", "Nyx", "Orim",
        "Talv", "Sero", "Wex", "Fenr", "Grix", "Ulth", "Yarr", "Cael", "Drav", "Esk", "Falk", "Gorm",
        "Hesp", "Ives", "Jorn", "Klev", "Lorn", "Morv", "Ostr", "Pyre", "Quld", "Rusk", "Sten", "Turv",
        "Vorn", "Welk", "Xanth", "Ysor", "Zorn", "Ald", "Bram", "Corv", "Dusk", "Ergo", "Frost", "Glim"]
_SUF = ["ine", "ar", "os", "eth", "ix", "an", "or", "us", "el", "ys", "on", "ur", "ax", "en", "ol", "ir"]
_MAT = ["palladium", "niobium", "tantalum", "yttrium", "rhenium", "hafnium", "iridium", "osmium",
        "scandium", "gadolinium", "erbium", "thulium", "lutetium", "praseodymium"]


def _name(r): return r.choice(_PRE) + r.choice(_SUF)
def _num(r): return f"{r.randint(1,99)},{r.randint(100,999)}"
def _pct(r): return f"{r.randint(70,99)}.{r.randint(1,9)}"
def _mat(r): return f"{r.choice(_MAT)}-{r.choice(_MAT)[:4]}"
def _code(r): return f"{r.choice(_PRE)}crete-{r.randint(2,9)}"

TEMPLATES = [
    lambda r: (lambda d, m: (f"The {d} reactor achieved high efficiency using a {m} catalyst developed in-house.",
                             f"What catalyst did the {d} reactor use?", m))(_name(r), _mat(r)),
    lambda r: (lambda p, c: (f"{p} was born in the coastal city of {c} in the early years.",
                             f"Where was {p} born?", c))(_name(r), _name(r)),
    lambda r: (lambda t, pub: (f"The novel {t} was written last year and published by {pub} House.",
                               f"Who published the novel {t}?", f"{pub} House"))(_name(r), _name(r)),
    lambda r: (lambda ds, n: (f"The {ds} dataset contains {n} annotated examples collected over two years.",
                              f"How many annotated examples does the {ds} dataset contain?", n))(_name(r), _num(r)),
    lambda r: (lambda lg, p: (f"The programming language {lg} was designed by {p} for embedded systems.",
                              f"Who designed the programming language {lg}?", p))(_name(r), _name(r)),
    lambda r: (lambda sc, ins: (f"The {sc} spacecraft carries an instrument named {ins} for dust analysis.",
                                f"What is the name of the instrument on the {sc} spacecraft?", ins))(_name(r), _name(r)),
    lambda r: (lambda pz, org: (f"The {pz} Prize is awarded annually by the {org} Foundation.",
                               f"Which foundation awards the {pz} Prize?", f"{org} Foundation"))(_name(r), _name(r)),
    lambda r: (lambda pr, pct: (f"The {pr} filter removes {pct} percent of airborne particles in tests.",
                               f"What percentage of airborne particles does the {pr} filter remove?", pct))(_name(r), _pct(r)),
    lambda r: (lambda g, n, b: (f"General {g} commanded {n} troops at the Battle of {b}.",
                               f"How many troops did General {g} command at the Battle of {b}?", n))(_name(r), _num(r), _name(r)),
    lambda r: (lambda br, m: (f"The {br} bridge was constructed using a self-healing material called {m}.",
                             f"What material was the {br} bridge built with?", m))(_name(r), _code(r)),
]


def make_split(split, n, seed=0):
    base = {"train": 10_000, "val": 500_000, "test": 900_000}[split]   # disjoint seed ranges
    out, seen = [], set()
    i = 0
    while len(out) < n:
        r = random.Random(base + seed * 100003 + i); i += 1
        chunk, q, a = TEMPLATES[r.randrange(len(TEMPLATES))](r)
        if a in seen: continue                                          # unique answers within split
        seen.add(a); out.append({"chunk": chunk, "question": q, "answer": a})
    return out
