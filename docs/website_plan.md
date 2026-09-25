# Project website — deployment plan

## How TeamCraft does it

<https://teamcraft-bench.github.io/> is a single static HTML page served by GitHub Pages. The
address follows GitHub's organization-site convention: an organization named `teamcraft-bench`
owns a repository named `teamcraft-bench.github.io`, and GitHub serves that repository at the
organization's root URL. The page has no build step and no framework. It carries the abstract,
task videos, method, results, the team, and links to the paper, code, data and BibTeX.

Our `site/` folder is already this kind of page: hand-written `index.html`, a generated
`gallery.html`, and static assets. It is 233 MB in 791 files, and no file is over 15 MB, which is
inside GitHub Pages' limits (1 GB per site, 100 MB per file, about 100 GB of traffic per month).

## During review: Anonymous GitHub (recommended)

Anonymous GitHub serves a mirrored repository's GitHub Pages site at
`https://anonymous.4open.science/w/<id>/`, next to the code mirror at `/r/<id>/`. Verified in its
source (`src/server/routes/webview.ts`, the "GitHub Pages" option of the anonymize form):

- **The source repository must have GitHub Pages enabled**, and Pages must publish from the same
  branch that is anonymised. The page root is the Pages folder, which GitHub restricts to `/`
  or `/docs`.
- A private source repository needs GitHub Pro (or Team) to enable Pages; Pro is free for
  students through GitHub Education. Note that Pages on a private Pro repository is still
  publicly reachable at `<username>.github.io/<repo>`, which names the account. Never link that
  address; share only the anonymous one.
- The page runs in a sandbox: scripts, popups and autoplay work, but it has an opaque origin.
  Our site uses neither `fetch` nor browser storage, so it works as is. External fonts and the
  KaTeX CDN load normally.
- Files up to 100 MB each; our largest is 15 MB.
- Text files pass through the term replacement, so the term list must only contain strings that
  never occur legitimately in the site (names, usernames, institution). Our scan finds none, so
  the list is a safety net.
- After 200 requests per visitor within 15 minutes, each further request is delayed (150 ms,
  rising to 5 s). The main page makes about 95 requests, so it is unaffected. The gallery makes
  up to 528 as a reader scrolls, so thin it with `keep.txt` before publishing.
- Mirrors expire on the date you set; pick one after the rebuttal period, or "never".

Steps:

1. Create a new **private** repository, for example `wire-site`, containing the contents of
   `site/` at its root, committed with a neutral identity:

   ```bash
   cp -r site ../wire-site && cd ../wire-site
   git init -b main
   git -c user.name="Anonymous" -c user.email="anonymous@example.com" add -A
   git -c user.name="Anonymous" -c user.email="anonymous@example.com" commit -m "WIRE project page"
   git remote add origin https://github.com/<you>/wire-site.git && git push -u origin main
   ```

2. In that repository: Settings → Pages → *Deploy from a branch*, branch `main`, folder `/ (root)`.
3. At anonymous.4open.science, anonymise `wire-site`: branch `main`, tick **GitHub Pages**, add
   the term list, set the expiry.
4. Open `https://anonymous.4open.science/w/<id>/` in a private window and click through the page,
   the gallery, the PDF and every link.
5. Link the page's "Code" button to the code mirror (`/r/<id>/`) and put both anonymous URLs in
   the paper's reproducibility statement.

The site stays in its own repository rather than in the code mirror because Pages can only
publish from `/` or `/docs`, and `docs/` already holds the code documentation.

## After acceptance

Move to the TeamCraft pattern: an organization site at `https://<org>.github.io/`. Creating the
organization now from a neutral account (below) also works during review and keeps one URL
throughout; Anonymous GitHub is the simpler option for review alone.

| | During review | After acceptance |
|---|---|---|
| URL | `https://<org>.github.io/` | same |
| Authors, affiliations, BibTeX | absent | added |
| Paper link | anonymous PDF on the site | arXiv / OpenReview |
| Code link | Anonymous GitHub mirror | public lab repository |
| Data link | OSF anonymised view-only link | Zenodo DOI |

## Why a separate account

GitHub shows who pushed to a public repository and who triggered each Pages deployment: commit
authors, and the Actions tab's "triggered by" line. A site pushed from a personal account reveals
that account, even if the page itself names nobody. So:

1. Create a new GitHub account with a neutral name and a fresh e-mail address.
2. From that account, create the organization, for example `wire-bench`. Check the name is free
   at `https://github.com/wire-bench`. Organization membership defaults to private; leave it so.
3. Only that account pushes to the site repository during review.

## Steps for the organization site

1. **Prepare the content** (also required before the Anonymous GitHub route) (in this repository, where `site/` is git-ignored):
   - Replace `site/assets/paper.pdf` with the v6 anonymous build. The current file has 46 pages;
     v6 has 48.
   - Thin the clip gallery: `site/assets/videos/candidates/` holds 701 files and has no
     `keep.txt` yet. Write the keep list, then rerun `analysis/make_site_gallery.py` (see
     `site/README.md`, "Thinning the gallery").
   - Point the Code and Data buttons at the anonymous mirror and the OSF link, or leave them as
     "on acceptance" (`data-needs-url`, already on five rows).
   - Run the anonymity scan below; it currently prints nothing.
2. **Create the site repository** `wire-bench/wire-bench.github.io` (public), from the neutral
   account.
3. **Publish** with one anonymous commit, so no history travels with it:

   ```bash
   cp -r site /tmp/wire-site && cd /tmp/wire-site
   git init -b main
   git -c user.name="WIRE" -c user.email="<neutral address>" add -A
   git -c user.name="WIRE" -c user.email="<neutral address>" commit -m "WIRE project page"
   git remote add origin https://github.com/wire-bench/wire-bench.github.io.git
   git push -u origin main
   ```

4. **Turn on Pages**: Settings → Pages → Source: *Deploy from a branch*, branch `main`, folder
   `/ (root)`. No workflow is needed for a static page. The existing
   `.github/workflows/pages.yml` in this repository is for the personal repository and is not
   part of the code release; do not copy it.
5. **Check** the live page in a private window: the page, the gallery, the PDF, and every
   outgoing link. Then put the URL in the paper's reproducibility statement.
6. **At acceptance**: add the team section, BibTeX and final links, invite the authors to the
   organization, and optionally attach a custom domain (Settings → Pages → Custom domain). The
   `github.io` address keeps working and redirects.

## Anonymity scan

Run before every push while under review. It must print nothing.

```bash
grep -rniwE "marcu|acmarcu|tudelft|delft|daic|tapri|anacmarcu|@gmail\.com|@tudelft\.nl" site \
    --include=*.html --include=*.js --include=*.css --include=*.json --include=*.md
grep -roE "github\.com/[A-Za-z0-9_.-]+" site --include=*.html | grep -v "github.com/mikelma"
```

Also open `paper.pdf` and confirm the header reads "Anonymous authors", and check that no video
frame or chat excerpt contains a username.
