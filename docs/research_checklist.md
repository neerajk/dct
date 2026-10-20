# Research checklist (one pass per campus)

Every answer becomes a row in `data/seed/claims.csv` (or a milestone) with a `source_id`.
Source tiers: 1 operator/government/regulator/filing, 2 planning portal/registry/major press,
3 trade press, 4 aggregator (Baxtel, Datacentres.com ...) = leads only.

## A. Identity
1. Official campus name and building IDs (e.g. AMS11, NAV1A)?
2. Developer/owner, operator, end tenant?
3. Program (announcement) it belongs to? Joint venture / partners?

## B. Location
4. Village / survey no. / industrial park / street address?
5. One plot or several? Area of each plot (ha or acres)?
6. Any published coordinates? (PARIVESH environmental-clearance forms often list them.)
7. OSM feature id?
8. Location precision: exact / parcel / locality / city?

## C. Capacity
9. IT MW in total and per phase? Number of buildings?
10. Grid MW sanctioned or applied for (often differs from IT MW)?

## D. Money
11. Investment announced (amount, currency, year of the statement)?
12. Lenders, bonds, state incentives, MoU?

## E. Announced timeline (each = one `milestones.csv` row)
13. Dates of: announcement, MoU, land allotment, clearance/permit, groundbreaking, go-live, per phase.
14. Any REVISED date? Keep the old row and add a new row with `is_revised=true`, `supersedes_id` set.

## F. Grid and utilities
15. Substation (name, kV, distance) and utility (APEPDCL, TGSPDCL, MSEDCL, Liander, TenneT, Syna ...)?
16. Water source? Backup-generator or gas-plant permits?

## G. Status
17. Latest dated progress report? Delay, court case, community opposition?

## Where to look
- India: PARIVESH, APEDB/SIPB minutes, TGIIC/MIDC allotments, Sify SEC 20-F/6-K, Adani and Anant Raj stock-exchange filings, electricity-regulator filings.
- NL: omgevingsloket, ruimtelijkeplannen.nl.
- DE: zoning-plan notices (Hanau, Dietzenbach, Offenbach), operator press releases.
- Aggregators only as leads; never the only source for a date or MW figure.
