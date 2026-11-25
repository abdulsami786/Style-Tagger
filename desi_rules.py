"""All the desi fashion knowledge in one place.

Occasion in desi fashion comes from fabric, embroidery, silhouette and price. Printed lawn is daily wear,
chiffon with dabka is a wedding outfit, a lehenga at Rs 400k is a bride's. This file turns a product into:

  signals(product)      named evidence ("formal fabric", "heavy handwork", "bridal silhouette", price band...)
  rule_scores(signals)  a 0-1 score per tag from that evidence; the model uses these as inputs and learns
                        from the hand labels how much each one is worth
  choose_tags(...)      the hard rules applied to the model's choice. Labels cannot override these;
                        change them here.

To add a tag: add it to TAGS, describe it in TAGS.md, label examples. A rule score is optional.
"""

import math
import re

TAGS = ["bridal_wear", "wedding", "festive_wear", "party_wear", "casuals", "smart_casual",
        "office_wear", "modest", "streetwear", "sportswear", "travel_wear"]


def rx(*words: str) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b")


WORDS = {
    # occasion words brands put in names and descriptions
    "cue_bridal": rx(r"brid(?:al|e)s?", r"walk down the aisle"),
    "cue_wedding": rx(r"wedding", r"shaa?di", r"bara+t", r"wa+lima", r"va+lima", r"nikk?ah", r"mehn?(?:d|e)i",
                      r"mayu+n", r"dholki", r"reception", r"engagement", r"groom", r"dulha", r"sangeet"),
    "cue_festive": rx(r"eid", r"festive", r"festivities", r"chand raat", r"celebrations?", r"dawat"),
    "cue_party": rx(r"party", r"evening", r"cocktail", r"glam(?:orous)?", r"soiree", r"dinner", r"gala", r"night out",
                    r"head[- ]turner", r"statement look"),
    "cue_casual": rx(r"casual", r"every ?day", r"daily", r"basic", r"essentials?", r"lounge", r"relaxed",
                     r"effortless", r"brunch", r"errands"),
    "cue_office": rx(r"office", r"work ?wear", r"professional", r"business", r"corporate", r"9 to 5"),
    "cue_formal": rx(r"formals?", r"semi[- ]formal", r"luxury", r"luxe", r"couture", r"premium", r"opulent", r"regal"),
    "cue_travel": rx(r"travel", r"airport", r"resort", r"vacation", r"holiday", r"beach"),
    # fabrics
    "fab_casual": rx(r"lawn", r"cotton", r"cambric", r"khaddar", r"khadar", r"linen", r"slub", r"viscose", r"marina",
                     r"dobby", r"seersucker", r"voile", r"jersey", r"fleece", r"denim", r"twill", r"poplin", r"cross ?hatch",
                     r"wash ?(?:&|n|and)? ?wear", r"terry", r"wool", r"cottel", r"grip"),
    "fab_formal": rx(r"chiffon", r"organza", r"net", r"silk", r"raw ?silk", r"sheesha silk", r"shisha silk", r"velvet",
                     r"jamawa+r", r"tissue", r"banarsi", r"masoori", r"maysuri", r"mesori", r"lama", r"lame", r"tulle",
                     r"satin", r"zari", r"katt?an", r"khaddi net", r"khadi net", r"georgette", r"brocade", r"karandi"),
    # embellishment
    "work_print": rx(r"printed", r"print", r"digital", r"block ?print(?:ed)?", r"screen ?print(?:ed)?", r"dyed", r"solid", r"plain"),
    "work_light": rx(r"embroider(?:ed|y)", r"embroided", r"thread ?work", r"aari", r"ari", r"schiffli", r"lace",
                     r"applique", r"chikan(?:kari)?", r"cutwork", r"pintucks?", r"embellish(?:ed|ments?)"),
    "work_heavy": rx(r"hand ?(?:work|worked|embellish(?:ed|ments?)|embroider(?:ed|y)|crafted)", r"zardozi", r"dabka",
                     r"dubka", r"tilla", r"tila", r"gott?a", r"sequins?", r"sequence", r"kora", r"mukesh(?:kari)?", r"cut ?da+na",
                     r"kat ?dana", r"crystals?", r"swarovski", r"pearls?", r"mirror", r"kundan", r"stones?", r"beads?",
                     r"beadwork", r"naqshi", r"nakshi", r"kamdani", r"resham", r"diamante", r"rhinestones?", r"heavily",
                     r"intricate", r"french knots?", r"badla"),
    # silhouettes and garments
    "g_bridal": rx(r"lehn?ga", r"lehenga", r"lenga", r"gharara", r"sharara", r"pishwa+s", r"peshwa+[sz]", r"kalidaa?r",
                   r"choli", r"farshi"),
    "g_groom": rx(r"sherwani", r"prince ?coat", r"princesuit"),
    "g_waistcoat": rx(r"waist ?coat", r"vest ?coat", r"koti"),
    "g_saree": rx(r"sa+ree", r"sari"),
    "g_gown": rx(r"gown", r"sequin dress", r"cocktail dress", r"evening dress", r"maxi dress", r"bodycon", r"corset",
                 r"one[- ]shoulder", r"fishtail", r"mermaid", r"jumpsuit"),
    "g_kaftan": rx(r"kaftan", r"caftan", r"poncho", r"cape"),
    "g_men_formal": rx(r"formal shirt", r"dress shirt", r"dress pants?", r"formal pants?", r"formal trousers?",
                       r"suiting", r"tuxedo", r"blazer", r"2 ?pc suit", r"2-piece coat"),
    "g_smart": rx(r"polo", r"chinos?", r"button[- ]down", r"cardigan", r"sweater", r"knit(?:ted)?", r"oxford",
                  r"co-?ord", r"cord set", r"smart fit", r"semi formal"),
    "g_street": rx(r"hoodies?", r"sweatshirts?", r"graphic", r"oversized", r"boxy", r"baggy", r"cargo", r"joggers?",
                   r"barrel fit", r"bomber", r"slogan", r"acid wash", r"tee"),
    "g_sport": rx(r"activewear", r"active wear", r"gym", r"yoga", r"track ?suit", r"athleisure", r"performance",
                  r"cardio", r"sports? bra", r"dri[- ]?fit", r"workout", r"athletic"),
    "g_modest": rx(r"abayas?", r"hijabs?", r"niqab", r"jilbab", r"namaz", r"prayer", r"chadar", r"burqa"),
    "g_lounge": rx(r"sleep ?wear", r"night ?(?:suit|wear|trousers?)", r"pj'?s", r"pyjamas?", r"lounge ?wear", r"cool max"),
    "g_western_basic": rx(r"t-?shirts?", r"tees?", r"jeans", r"denim", r"shorts", r"tank ?top", r"tights", r"leggings?",
                          r"crop top", r"skirt"),
    "g_eastern": rx(r"kurta", r"kurti", r"kameez", r"qameez", r"shalwar", r"salwar", r"dupatt?a", r"duppatta", r"suit",
                    r"3[- ]?pc", r"2[- ]?pc", r"3 piece", r"2 piece", r"unstitched", r"angh?rakha", r"anarkali",
                    r"izaa?r", r"pajama", r"pyjama", r"culottes?", r"trouser"),
    "g_accessory": rx(r"earrings?", r"jhumk[ia]s?", r"rings?", r"necklace", r"bangles?", r"clutch", r"sunglasses",
                      r"belt", r"heels", r"flats", r"sneakers", r"fragrance", r"perfume", r"hair clip", r"bag", r"jewel",
                      r"sandals?", r"slippers?", r"cap"),
}

# The shop's own category (normalised) pins a garment family regardless of the text
SUBSUB_FAMILY = {
    "lehenga": "g_bridal", "gharara": "g_bridal", "sharara": "g_bridal", "pishwas": "g_bridal", "peshwas": "g_bridal",
    "sherwani": "g_groom", "princecoat": "g_groom", "saree": "g_saree", "gown": "g_gown", "jumpsuit": "g_gown",
    "abayas": "g_modest", "abaya": "g_modest", "hijab": "g_modest", "jilbab": "g_modest", "niqab": "g_modest",
    "prayerchadar": "g_modest", "hoodies": "g_street", "sweatshirt": "g_street", "sweatshirts": "g_street",
    "sleepwear": "g_lounge", "polos": "g_smart", "blazer": "g_men_formal", "suits": "g_men_formal",
    "tshirts": "g_western_basic", "jeans": "g_western_basic", "shorts": "g_western_basic", "tights": "g_western_basic",
    "tanktop": "g_western_basic", "waistcoat": "g_waistcoat", "koti": "g_waistcoat", "kaftan": "g_kaftan",
}

# "perfect for a bridal shower" / "the bride-to-be" describe a guest, not the bride
NOT_BRIDAL = re.compile(r"bridal showers?|bride[- ]to[- ]be|bridesmaids?|bridal occasions|(?:sister|mother) of the bride")
BRIDE_PRICE = 250_000     # a bridal silhouette, or a listing with no description, above this is the bride's outfit
BRIDE_MIN_PRICE = 40_000  # below this a "bridal" word is a design name or a guest outfit, not the bride's

PRICE_BANDS = [(4_000, 0.0), (10_000, 0.3), (25_000, 0.7), (60_000, 1.2), (150_000, 1.7), (math.inf, 2.2)]
URL_HOST = re.compile(r"^https?://[^/]+")

# Plain names for the signals shown as "why" in the web app, strongest evidence first
SIGNAL_NAMES = {
    "cue_bridal": "says bridal / bride", "cue_wedding": "wedding words (mehndi, barat, nikah...)",
    "g_groom": "sherwani / prince coat", "g_bridal": "bridal silhouette (lehenga, gharara, pishwas...)",
    "cue_festive": "festive / Eid words", "cue_party": "party / evening words", "g_gown": "gown / cocktail dress",
    "g_saree": "saree", "work_heavy": "heavy handwork (zardozi, dabka, sequins...)",
    "fab_formal": "formal fabric (chiffon, organza, silk...)", "work_light": "embroidery",
    "cue_formal": "luxury / formal words", "g_waistcoat": "waistcoat", "g_modest": "abaya / hijab / prayer wear",
    "modest_cat": "Modest Wear category", "g_sport": "activewear words",
    "g_street": "streetwear (hoodie, graphic, oversized...)", "g_men_formal": "formal shirt / suit / blazer",
    "formal_shirt_men": "men's formal shirt or trousers", "cue_office": "office / work words",
    "g_smart": "polo / chinos / knit / co-ord", "g_lounge": "sleep / lounge wear", "cue_travel": "travel / resort words",
    "fab_casual": "everyday fabric (lawn, cotton, khaddar...)", "work_print": "printed / dyed / plain",
    "cue_casual": "casual / everyday words", "g_western_basic": "tee / jeans / shorts",
}


def url_words(r: dict) -> str:
    # brands encode collections and colours in URL slugs and image file names ("/Formal/", "Mina-bridals-2.jpg")
    return " ".join(re.sub(r"[-_/.]+", " ", URL_HOST.sub("", u or "")) for u in (r.get("url"), r.get("imageUrl")))


def product_text(r: dict) -> str:
    return " ".join([r["name"], r["desc"], r["details"], r["keywords"], url_words(r)]).lower()


def signals(r: dict) -> dict[str, float]:
    """Named evidence, each 0/1 or a small number. Same keys for every product."""
    text = product_text(r)
    name = r["name"].lower()
    s = {k: float(bool(p.search(text))) for k, p in WORDS.items()}
    s["cue_bridal"] *= not NOT_BRIDAL.search(text)
    # a garment named in the title is stronger than one mentioned in a styling tip
    for k in ("g_bridal", "g_groom", "g_saree", "g_gown", "g_street", "g_sport", "g_modest", "g_accessory"):
        s[k + "_name"] = float(bool(WORDS[k].search(name)))
    fam = SUBSUB_FAMILY.get(r["subSub"])
    if fam:
        s[fam] = 1.0
    s["n_heavy"] = min(len(set(WORDS["work_heavy"].findall(text))), 6) / 6
    s["men"] = float(r["gender"] == "men")
    s["western"] = float(r["subCategory"] == "Western Wear")
    s["modest_cat"] = float(r["subCategory"] == "Modest Wear")
    s["unstitched"] = float(r["subCategory"] == "Unstitched")
    s["no_text"] = float(not r["desc"] and len(name.split()) <= 3)
    s["formal_shirt_men"] = float(s["men"] and bool(re.search(r"formal|dress shirt|business|office", text))
                                  and bool(re.search(r"shirt|pant|trouser", name)))
    price = r["price"] or r["brandMedian"] or 0
    s["log_price"] = math.log10(price) if price > 0 else 3.5
    s["log_brand_median"] = math.log10(r["brandMedian"]) if r["brandMedian"] else 3.5
    s["price_pct"] = r["pricePct"] if r["pricePct"] is not None else 0.5
    s["price_band"] = next(v for cap, v in PRICE_BANDS if price < cap)
    apparel = not (s["g_accessory_name"] or s["g_western_basic"] or s["g_street"] or s["g_sport"])
    # price alone marks a bride only for a bridal silhouette or a text-less couture listing;
    # couture formals (Sana Safinaz D-series at Rs 300-900k) are wedding/party wear, not the bride's
    s["bride"] = float(not s["men"] and apparel and (
        (s["cue_bridal"] and price >= BRIDE_MIN_PRICE)
        or (price >= BRIDE_PRICE and bool(s["g_bridal"] or s["no_text"]))))
    return s


def formality(s: dict) -> float:
    """1 daily .. 5 bridal, from fabric, work, silhouette and price."""
    f = 1.0 + s["price_band"]
    f += 0.6 * s["fab_formal"] - 0.3 * s["fab_casual"] * (1 - s["fab_formal"])
    f += 0.4 * s["work_light"] + 0.9 * s["work_heavy"] + 0.6 * s["n_heavy"]
    f -= 0.4 * s["work_print"] * (1 - s["work_heavy"])
    f += 0.8 * s["g_bridal"] + 1.0 * s["g_groom"] + 0.4 * s["g_saree"] + 0.5 * s["g_gown"]
    f += 0.5 * s["cue_wedding"] + 0.3 * s["cue_festive"] + 0.3 * s["cue_formal"] - 0.5 * s["cue_casual"]
    f = 5.0 if s["bride"] else min(f, 4.6)  # 5 is reserved for the bride's own outfit
    if s["g_street_name"] or s["g_sport_name"]:
        f = min(f, 1.5)
    return max(1.0, min(5.0, f))


def rule_scores(s: dict) -> dict[str, float]:
    """Each tag's rule score in [0, 1]. A tag with no rule here scores 0 and is learned from labels alone."""
    f = formality(s)
    eastern = s["g_eastern"] or s["g_bridal"] or s["g_groom"] or s["g_waistcoat"] or s["unstitched"]
    western = s["western"] and not eastern
    sporty = max(s["g_sport"], s["g_sport_name"])
    street = max(s["g_street"], s["g_street_name"]) * (1 - sporty * 0.5)
    modest = max(s["g_modest"], s["modest_cat"])
    office = max(s["g_men_formal"] * s["men"], s["formal_shirt_men"], s["cue_office"],
                 # women's solid / minimal cotton kurta reads as office wear
                 float(not s["men"] and eastern and s["fab_casual"] and not s["work_heavy"] and f < 2.2
                       and bool(s["work_print"]) and not s["fab_formal"]) * 0.4)

    def band(lo: float, hi: float) -> float:
        """1 inside [lo, hi], fading out linearly over one formality step."""
        return max(0.0, 1 - max(lo - f, f - hi, 0))

    sc = {
        "bridal_wear": s["bride"],
        "wedding": max(band(3.6, 5), s["g_groom"], s["cue_wedding"] * 0.8,
                       s["g_waistcoat"] * s["men"] * 0.5) * (1 - street),
        "festive_wear": max(band(2.0, 3.6) * float(bool(eastern or s["g_kaftan"])), s["cue_festive"],
                            s["g_waistcoat"] * 0.7),
        "party_wear": max(band(2.6, 5) * float(bool(western or s["g_gown"] or s["g_saree"] or s["g_kaftan"])),
                          s["g_gown"], s["cue_party"] * 0.7, s["g_saree"] * 0.6),
        # a specific style (street, sport, modest, office) outranks the generic daily-wear tag
        "casuals": max(band(1, 1.9) * (1 - 0.5 * office - 0.2 * street - 0.2 * sporty - 0.3 * modest),
                       s["cue_casual"] * 0.6, s["g_western_basic"] * 0.7, street * 0.6),
        "smart_casual": max(s["g_smart"] * band(1, 2.6), s["g_men_formal"] * (1 - s["formal_shirt_men"]) * 0.4),
        "office_wear": office,
        "modest": modest,
        "streetwear": street,
        "sportswear": sporty,
        "travel_wear": max(s["g_lounge"], s["cue_travel"]) * 0.7,
    }
    if s["g_accessory_name"] and not eastern:
        sc = {t: v * 0.3 for t, v in sc.items()}
        sc["casuals"] = max(sc["casuals"], 0.6)
    return {t: sc.get(t, 0.0) for t in TAGS}


def evidence(s: dict, limit: int = 6) -> list[str]:
    """The signals present for this product, strongest first (keys of SIGNAL_NAMES)."""
    return [k for k in SIGNAL_NAMES if s.get(k)][:limit]


def choose_tags(probs: dict[str, float], thresholds: dict[str, float], s: dict,
                max_tags: int = 3) -> tuple[list[str], list[dict]]:
    """The model's tags (probability above that tag's threshold, most likely first), then the hard rules.

    Returns (tags, notes). Each note says which rule added or removed a tag, so the web app can show it.
    """
    order = sorted(probs, key=probs.get, reverse=True)
    tags = [t for t in order if probs[t] >= thresholds[t]] or order[:1]
    notes = []

    def note(tag: str, change: str, rule: str) -> None:
        notes.append({"tag": tag, "change": change, "rule": rule})

    # 1. bridal_wear comes only from the bride rule; menswear is never bridal (groom wear is `wedding`)
    if "bridal_wear" in tags:
        tags.remove("bridal_wear")
        if not s["bride"]:
            note("bridal_wear", "removed", "menswear is never bridal" if s["men"] else "the bride rule did not fire")
    elif s["bride"]:
        note("bridal_wear", "added", "bride rule")
    if s["bride"]:
        tags.insert(0, "bridal_wear")  # a bride's outfit leads with bridal_wear
    # 2. a bride's outfit is also wedding wear
    if "bridal_wear" in tags and "wedding" not in tags:
        tags.insert(1, "wedding")
        note("wedding", "added", "a bride's outfit is also wedding wear")
    # 3. daily wear and wedding wear sit at opposite ends of formality: keep the more confident one
    if "casuals" in tags and "wedding" in tags:
        drop = "casuals" if probs["wedding"] >= probs["casuals"] or "bridal_wear" in tags else "wedding"
        tags.remove(drop)
        note(drop, "removed", "casuals and wedding never appear together")
    # 4. modest needs an explicit modest garment or the Modest Wear category, not just a long silhouette
    if "modest" in tags and not (s["g_modest"] or s["modest_cat"]):
        tags.remove("modest")
        note("modest", "removed", "no abaya / hijab-type garment and not in the Modest Wear category")
    blocked = {"bridal_wear", "modest"}
    return tags[:max_tags] or [t for t in order if t not in blocked][:1], notes


# ---- the bridal tagger (app.BridalApp): which of her wedding functions a bride's outfit is for --------------------
# The app's bridal substyles, in its order and spelling. They are not TAGS: only a bride's outfit gets one.
EVENTS = ["mehendi", "barat", "valima", "nikah", "dholki"]
# Style's bridal_wear is the bride's heaviest outfits. A wedding outfit in a bridal silhouette (lehenga, gharara,
# sharara, pishwas...) from this price is one for her lighter functions too; below it they were mostly guests' outfits.
BRIDE_OUTFIT_PRICE = 100_000
BRIDE_MEHNDI_PRICE = 70_000   # the same in the mehndi colours: the bride dresses lighter for her mehndi and dholki

# the function a listing names ("mehndi green" is a colour, not the function)
EVENT_WORDS = {
    "mehendi": rx(r"meh[ea]?n?di(?![ -]?green)", r"may(?:u+|oo?)n", r"haldi", r"ubtan"),
    "barat": rx(r"bara+t"),
    "valima": rx(r"wa+lima", r"va+lima", r"reception"),
    "nikah": rx(r"nikk?ah"),
    "dholki": rx(r"dholki", r"dholak", r"sangeet"),
}
# Otherwise its main colour, from the bride's palette for each function: red and maroon for the barat; yellow, orange,
# green, hot pink and multicolour for the mehndi and dholki; white and ivory for the nikah; gold, silver and champagne
# for the valima; pastels for the nikah or valima. Pink, blue and purple are worn to any of them. Brides don't wear black.
EVENT_COLOURS = [
    (rx(r"red", r"maroon", r"burgundy", r"crimson", r"scarlet", r"wine", r"oxblood"), ["barat"]),
    (rx(r"yellow", r"mustard", r"lemon", r"marigold", r"saffron", r"orange", r"tangerine", r"rust", r"green",
        r"parrot", r"lime", r"pista(?:chio)?", r"multi[- ]?colou?r(?:ed)?", r"colou?rful", r"hot pink", r"shocking pink",
        r"rani pink", r"fuchsia", r"fuschia", r"magenta", r"turquoise", r"teal", r"feroz[ai]"), ["mehendi", "dholki"]),
    (rx(r"white", r"off[- ]?white", r"ivory", r"cream", r"pearl white"), ["nikah"]),
    (rx(r"gold(?:en)?", r"silver", r"champagne", r"bronze", r"metallic", r"beige", r"nude", r"fawn"), ["valima"]),
    (rx(r"pastels?", r"ice ?blue", r"powder ?blue", r"sky ?blue", r"light blue", r"baby ?pink", r"blush", r"tea ?pink",
        r"light pink", r"lilac", r"lavender", r"mint", r"peach"), ["nikah", "valima"]),
    (rx(r"black"), []),
]
MEHNDI_COLOURS = EVENT_COLOURS[1][1]


def event_text(p: dict) -> str:
    return " ".join([p["name"], url_words(p), p["desc"], p["details"], p["keywords"]]).lower()


def main_colour(p: dict) -> list[str] | None:
    """The functions a product's main colour points to: the colour named first, in its name, URL, then description.
    [] for black, None when it names no colour."""
    text = event_text(p)
    found = [(m.start(), events) for pat, events in EVENT_COLOURS if (m := pat.search(text))]
    return min(found, key=lambda x: x[0])[1] if found else None


def bridal_scope(p: dict, r: dict) -> bool:
    """Is product p a bride's outfit, for one of her functions? r is the style tagger's output for it. Yes when Style
    tags it bridal_wear or its model chose bridal_wear (only the bride rule's price removed it), though under the
    outfit price only in a bridal silhouette: a cheaper kurta "for the bride or a close attendee" is marketing. Yes too
    for a wedding outfit in a bridal silhouette at a bride's price. Never a black outfit, nor a sherwani or prince coat
    (one brand files its groom wear under women). A Style label decides: without bridal_wear it is a guest's outfit."""
    if r["gender"] != "women" or SUBSUB_FAMILY.get(p["subSub"]) == "g_groom" or WORDS["g_groom"].search(p["name"].lower()):
        return False
    colour = main_colour(p)
    if colour == []:
        return False
    price = p["price"] or 0
    silhouette = bool(SUBSUB_FAMILY.get(p["subSub"]) == "g_bridal" or WORDS["g_bridal"].search(product_text(p)))
    if "bridal_wear" in r["tags"] or any(n["tag"] == "bridal_wear" for n in r["ruleNotes"]):
        return silhouette or not 0 < price < BRIDE_OUTFIT_PRICE   # couture listings without a price are the bride's
    if r.get("source") == "label":
        return False
    floor = BRIDE_MEHNDI_PRICE if colour == MEHNDI_COLOURS else BRIDE_OUTFIT_PRICE
    return "wedding" in r["tags"] and price >= floor and silhouette


def bridal_events(p: dict, formality: float) -> list[str]:
    """A starting guess at the functions a bride's outfit is for: the one or two its listing names, else the ones its
    main colour points to. [] when neither."""
    named = [e for e, pat in EVENT_WORDS.items() if pat.search(event_text(p))]
    if 1 <= len(named) <= 2:   # a listing that names three or more is "for every function", which says nothing
        return named
    events = main_colour(p) or []
    # the dholki is the lighter function: the bride's heaviest outfits (formality 5, Style's bridal_wear) are for the mehndi
    return ["mehendi"] if "dholki" in events and formality >= 5 else events
