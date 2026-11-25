# Desi occasion tags

What each tag means. Label products by these definitions, so the tagger learns one consistent meaning per tag.

One product gets **1 to 3 tags**, ordered by strength. The first tag is the primary occasion.
The keys match the old system, so Algolia facets keep working.

Each product also gets a **formality** level from 1 to 5. Pakistani brands sell by these tiers:

| formality | Meaning | Typical cues |
|---|---|---|
| 1 daily | Wear it any day | Printed or basic lawn, cotton, khaddar, cambric, linen. Tees, jeans, hoodies. Price in the bottom third of the brand |
| 2 semi-formal | A step above daily | Light embroidery (neckline, daaman), dyed with embroidered patches, "luxury pret" daywear, formal shirts, chinos |
| 3 festive / formal | Eid, dawat, dinners | Embroidered 3-piece with a chiffon, organza or net dupatta. Jacquard, karandi, raw silk with medium work. Men's embroidered kurta, waistcoat sets |
| 4 wedding formal | Mehndi, barat, walima, nikah guest | Heavy embroidery or handwork (zardozi, dabka, tilla, gota, sequins, mirror), chiffon, organza, net, tissue, velvet, jamawar. Lehenga, gharara, sharara, pishwas. Sherwani, prince coat |
| 5 bridal | The bride's own outfit | "Bridal" or "bride" in the text, or couture lehenga, gharara or maxi with the heaviest work at the top of a couture house's price range |

## Tags

| Tag | Use when | Do not use when |
|---|---|---|
| `bridal_wear` | The **bride** would wear it: barat, walima, nikah or mehndi bride. Formality 5 | It is groom wear (that is `wedding`) or a guest outfit |
| `wedding` | Wedding functions as a guest, family or groom. Formality 4 or more, or formality 3 with formal fabric and heavy work. Includes groom sherwani and prince coat | It is a plain or lightly embroidered lawn suit |
| `festive_wear` | Eid, chand raat, family dawat, festive collections. Formality 2 to 3 eastern wear. Men's embroidered kurta or kurta pajama | It is plain daily wear or western wear without festive cues |
| `party_wear` | Evening events, dinners and parties, mostly western or fusion: gowns, sequin or embellished dresses, cocktail wear, glam co-ords, jumpsuits, party sarees. Can sit alongside `wedding` for evening formals | It is a traditional daytime suit |
| `casuals` | Everyday wear. Eastern: printed, dyed or basic lawn, cotton, khaddar, cambric, linen kurta sets, kurtis, shalwars, unstitched printed suits. Western: tees, jeans, casual shirts, sweaters, shorts | It has formal fabric or heavy work |
| `smart_casual` | Polished casual: polos, chinos, linen or neat button-downs, knit tops, tidy co-ords, solid kurtas with minimal detail, cardigans, casual blazers | It is a graphic tee or an athletic piece |
| `office_wear` | Workwear. Men: formal or dress shirts, dress trousers, suits, blazers. Women: solid or minimally embroidered kurta and trouser in cotton, lawn, cambric or linen, tunics, cigarette pants, formal trousers, workwear co-ords | It is heavy work, sheer, glam or streetwear |
| `modest` | Explicit modest garments: abaya, hijab, niqab, jilbab, prayer chadar, scarves and stoles sold as modest wear, and the Modest Wear subcategory. A kaftan counts only when it is a loose full-length cover or sold as modest wear | It is a regular 3-piece suit (otherwise every suit would qualify), or a designer kaftan-cut kurta or party kaftan |
| `streetwear` | Hoodies, sweatshirts, graphic or oversized tees, cargos, baggy or wide jeans, joggers, bombers | It is a plain polo or a formal shirt |
| `sportswear` | Activewear, gym, tracksuits, athletic leggings, sports bras, dri-fit | It is fashion leggings or tights worn under kurtas (`casuals`) |
| `travel_wear` | Comfort-first pieces: loungewear, sleepwear, relaxed jogger sets, light layering jackets. Always a secondary tag | Never use it as the only tag unless the item is pure loungewear |

## Edge cases

- **Unstitched fabric** gets the tags of the outfit it becomes. Printed lawn 3-piece unstitched is `casuals`. Embroidered chiffon unstitched is `festive_wear` plus `wedding`.
- **Dupattas, shawls and stoles** follow their fabric and work. A printed lawn dupatta is `casuals`. An embroidered organza dupatta is `festive_wear` plus `wedding`. A wool shawl is `casuals`.
- **Hijabs** are `modest`, plus `party_wear` when metallic or embellished.
- **Jewellery** is tagged by the occasion the text names. With no cue it is `festive_wear`. **Fragrance, belts and other basics** are `casuals`. These are not apparel.
- **Brand `subSubCategory` is a hint only.** Designer houses file full ensembles under "Dupatta" or "Pant". Text and image win.
- **Kids' items** follow the same rules.
- **Men's waistcoat over kurta** is `festive_wear` plus `wedding`. A western vest is `office_wear` plus `smart_casual`.
- **Men's unstitched suiting fabric** (wash and wear, boski, cotton) is `casuals` plus `office_wear`. Jamawar or karandi suiting is `festive_wear`.

## Bridal substyles

The Bridal tagger gives each **bride's outfit** 1 to 3 of these: the wedding functions she would wear it to, the app's
bridal substyles. Guests' outfits get none, however formal: a 3-piece suit, a kurta or a sister-of-the-bride lehenga is
not a bride's outfit. Label by the function a bride would wear the outfit to.

| Tag | The function | What the bride wears |
|---|---|---|
| `mehendi` | The mehndi (and mayun) night: henna, dancing, colour | Yellow, mustard, orange, green, hot pink or multicolour; gota, mirror work, phulkari; lehenga, gharara, sharara. Light to medium work |
| `barat` | The wedding day, when the bride leaves with the groom's family | Red, maroon or wine; her heaviest outfit (zardozi, dabka, tilla, stones); lehenga-choli or farshi gharara |
| `valima` | The groom's family's reception after the wedding | Pastels, silver, gold or champagne; elegant, medium-heavy work; gown, maxi or pastel lehenga |
| `nikah` | The marriage ceremony | White, off-white, ivory or soft pastels; understated work; maxi, anarkali, sharara or gharara, farshi |
| `dholki` | The informal music nights before the wedding | The mehndi palette, lighter: gharara or sharara with a short shirt, or a light lehenga; light gota and mirror work |

- **A pastel outfit** often suits both `nikah` and `valima`: give it both.
- **`dholki` is the lighter end of the mehndi palette.** A bright outfit can be both; her heaviest bright lehenga is `mehendi` only.
- **Pink, blue and purple** are worn to every function. Decide by the work and silhouette: the heaviest lehenga is `barat`, a glamorous gown is `valima`.
- **Black** is never a bride's outfit.
