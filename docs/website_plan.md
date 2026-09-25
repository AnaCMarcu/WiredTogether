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

## Recommendation

Use the same pattern under a neutral organization created now, so the URL survives from review to
camera-ready and only the content changes at acceptance.

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

Anonymous GitHub (anonymous.4open.science) can also serve a mirrored repository's static pages,
which is a fallback if a separate account is not wanted. Its URLs are opaque and expire, so it is
less suitable as the long-term address.

## Steps

1. **Prepare the content** (in this repository, where `site/` is git-ignored):
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
