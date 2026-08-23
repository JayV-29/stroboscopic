"""Presentation images beyond matplotlib: CC-licensed photographs of the actual
technology (PPG front-ends, wrist wearables, Cortex-M MCUs) fetched from Wikimedia
Commons with attribution, plus PIL-composed infographics (hero card, device
"energy budget" card, results contact sheet).

Photos need internet (Kaggle: Settings -> Internet on).  Without it the
composites are still produced, just without the photo panels.
"""
from __future__ import annotations

import io
import json
import os
import textwrap
import time
import urllib.parse
import urllib.request

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .viz import SURFACE, PAGE, INK, INK2, MUTED, SERIES, SEQ_BLUE

OK_LICENSES = ("cc0", "public domain", "cc by", "cc-by", "cc by-sa", "cc-by-sa", "pd")

PHOTO_QUERIES = {
    "ppg_sensor":    "pulse oximeter",
    "smartwatch":    "smartwatch heart rate sensor",
    "mcu":           "microcontroller chip",
    "accelerometer": "accelerometer",
    "ecg":           "electrocardiogram electrodes",
}


def _font(size: int, bold: bool = False):
    cands = ["/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
    for c in cands:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size, index=1 if (bold and c.endswith(".ttc")) else 0)
            except Exception:
                try:
                    return ImageFont.truetype(c, size)
                except Exception:
                    pass
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# Wikimedia Commons fetch
# --------------------------------------------------------------------------- #
def fetch_commons_photo(query: str, out_path: str, width: int = 1400, timeout: int = 20) -> dict | None:
    """Search Commons for a freely licensed photo, save it, return attribution dict."""
    api = ("https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6"
           f"&gsrsearch={urllib.parse.quote(query)}&gsrlimit=20&prop=imageinfo"
           f"&iiprop=url|extmetadata|size&iiurlwidth={width}&format=json")
    req = urllib.request.Request(api, headers={"User-Agent": "strobo-notebook/0.1 (research; contact via repo)"})
    data = None
    for attempt in range(4):                       # polite retry on 429 / transient errors
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            break
        except Exception as e:
            if attempt == 3:
                return None
            time.sleep(3.0 * (attempt + 1))
    pages = data.get("query", {}).get("pages", {})
    for p in sorted(pages.values(), key=lambda x: x.get("index", 99)):
        ii = (p.get("imageinfo") or [{}])[0]
        meta = ii.get("extmetadata", {})
        lic = meta.get("LicenseShortName", {}).get("value", "").lower()
        if not any(k in lic for k in OK_LICENSES):
            continue
        if ii.get("width", 0) < 600 or ii.get("mime", "image/jpeg") not in ("image/jpeg", "image/png"):
            continue
        url = ii.get("thumburl") or ii.get("url")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "strobo-notebook/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                img = Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception:
            continue
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        img.save(out_path, quality=92)
        return {"file": os.path.basename(out_path), "title": p.get("title", ""), "query": query,
                "license": meta.get("LicenseShortName", {}).get("value", ""),
                "artist": _strip_html(meta.get("Artist", {}).get("value", "")),
                "credit": _strip_html(meta.get("Credit", {}).get("value", "")),
                "source": ii.get("descriptionurl", "")}
    return None


def _strip_html(s: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", s or "").strip()


def gather_photos(out_dir: str, queries: dict | None = None, log=print) -> dict:
    """Download one CC photo per query into out_dir/photos and write ATTRIBUTION.md."""
    queries = queries or PHOTO_QUERIES
    got = {}
    pdir = os.path.join(out_dir, "photos")
    os.makedirs(pdir, exist_ok=True)
    for key, q in queries.items():
        info = fetch_commons_photo(q, os.path.join(pdir, f"{key}.jpg"))
        time.sleep(1.5)
        if info:
            got[key] = info
            log(f"photo {key}: {info['title']} [{info['license']}]")
        else:
            log(f"photo {key}: not available (no internet or no free result)")
    with open(os.path.join(pdir, "ATTRIBUTION.md"), "w") as f:
        f.write("# Photo attribution (Wikimedia Commons)\n\n")
        for k, v in got.items():
            f.write(f"- **{v['file']}** — {v['title']} — {v['artist'] or v['credit']} — {v['license']} — {v['source']}\n")
        if not got:
            f.write("No photos were downloaded (internet disabled or no free-licence result).\n")
    return got


# --------------------------------------------------------------------------- #
# PIL composites
# --------------------------------------------------------------------------- #
def _rounded(img: Image.Image, radius: int = 28) -> Image.Image:
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius, fill=255)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def _fit(img: Image.Image, w: int, h: int) -> Image.Image:
    s = max(w / img.width, h / img.height)
    im = img.resize((int(img.width * s) + 1, int(img.height * s) + 1), Image.LANCZOS)
    x, y = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x, y, x + w, y + h))


def _shadow(canvas: Image.Image, box, radius=28, blur=18, alpha=70):
    x0, y0, x1, y1 = box
    sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([x0 + 6, y0 + 10, x1 + 6, y1 + 10], radius, fill=(0, 0, 0, alpha))
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    canvas.alpha_composite(sh)


def _contain(img: Image.Image, w: int, h: int, bg=SURFACE) -> Image.Image:
    s = min(w / img.width, h / img.height)
    im = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
    out = Image.new("RGB", (w, h), bg)
    out.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return out


def _paste_card(canvas: Image.Image, img: Image.Image, box, radius=28, contain=False):
    """Photos are cover-cropped; figures (contain=True) are letter-boxed so nothing is cut off."""
    x0, y0, x1, y1 = box
    _shadow(canvas, box, radius)
    fitted = _contain(img, x1 - x0, y1 - y0) if contain else _fit(img, x1 - x0, y1 - y0)
    canvas.alpha_composite(_rounded(fitted, radius), (x0, y0))


def hero_card(out_path: str, stats: dict, photo: str | None = None, curve_png: str | None = None,
              title: str = "Stroboscopic sensing", subtitle: str = "a 20 k-parameter world model decides when to wake the PPG"):
    """1920x1080 title card: photo (if any) + headline numbers + the primary curve."""
    W, H = 1920, 1080
    c = Image.new("RGBA", (W, H), PAGE)
    d = ImageDraw.Draw(c)
    # accent band
    d.rectangle([0, 0, W, 14], fill=SERIES["ours"])
    d.text((80, 70), title, font=_font(72, True), fill=INK)
    d.text((80, 160), subtitle, font=_font(34), fill=INK2)
    # stat tiles
    x, y = 80, 250
    n = max(1, len(stats))
    tw = (W - 160 - 24 * (n - 1)) // n
    for label, val in stats.items():
        d.rounded_rectangle([x, y, x + tw, y + 150], 22, fill=SURFACE, outline="#e1e0d9")
        vf = _font(46 if len(str(val)) <= 9 else 34, True)
        d.text((x + 24, y + 24), str(val), font=vf, fill=SERIES["ours"])
        d.text((x + 24, y + 100), label, font=_font(19), fill=INK2)
        x += tw + 24
    if photo and os.path.exists(photo):
        _paste_card(c, Image.open(photo).convert("RGB"), (80, 440, 860, 1000))
        d.text((80, 1010), "photo: Wikimedia Commons, see photos/ATTRIBUTION.md", font=_font(18), fill=MUTED)
        cx0 = 900
    else:
        cx0 = 80
    if curve_png and os.path.exists(curve_png):
        _paste_card(c, Image.open(curve_png).convert("RGB"), (cx0, 440, W - 80, 1000), contain=True)
    c.convert("RGB").save(out_path, quality=94)
    return out_path


def device_card(out_path: str, cost: dict, energy: dict, photos: dict):
    """1920x1080 'what runs on the device' card: MCU + IMU + PPG photos with the budget numbers."""
    W, H = 1920, 1080
    c = Image.new("RGBA", (W, H), PAGE)
    d = ImageDraw.Draw(c)
    d.rectangle([0, 0, W, 14], fill=SERIES["fixed_rate"])
    d.text((80, 60), "What runs on the device", font=_font(64, True), fill=INK)
    d.text((80, 140), "always-on IMU  ->  ~8 k MACs per tick  ->  a 125 ms PPG burst only when the cycle is informative", font=_font(28), fill=INK2)
    cols = [("mcu", "Cortex-M class MCU", [f"{cost.get('params', 0)/1000:.1f} k params (int8: {cost.get('int8 weight bytes', 0)/1024:.0f} kB)",
                                            f"{cost.get('MACs/tick (amortised)', 0):,} MACs / tick",
                                            f"~{cost.get('est. µs/tick on Cortex-M4 @48 MHz', 0):.0f} µs / tick,  {cost.get('duty at 32 Hz (%)', 0):.2f} % duty"]),
            ("accelerometer", "Always-on wrist IMU", ["32 Hz, 3 axes", f"~{energy.get('uj_per_imu_tick', 1.1)} µJ per tick", "drives the oscillator bank"]),
            ("ppg_sensor", "PPG front-end (duty-cycled)", ["8 samples = 125 ms per burst", f"~{energy.get('uj_per_burst', 135)} µJ per burst",
                                                             "fired by phase, not by a clock"])]
    x = 80
    for key, head, lines in cols:
        box = (x, 230, x + 560, 700)
        p = photos.get(key)
        if p and os.path.exists(p):
            _paste_card(c, Image.open(p).convert("RGB"), box)
        else:
            d.rounded_rectangle(box, 28, fill=SURFACE, outline="#e1e0d9")
            d.text((x + 30, 440), "(photo needs internet)", font=_font(24), fill=MUTED)
        d.text((x, 730), head, font=_font(34, True), fill=INK)
        for i, l in enumerate(lines):
            d.text((x, 790 + 44 * i), l, font=_font(26), fill=INK2)
        x += 600
    c.convert("RGB").save(out_path, quality=94)
    return out_path


def contact_sheet(out_path: str, pngs: list[str], title: str = "All figures", cols: int = 3, width: int = 2400):
    imgs = [Image.open(p).convert("RGB") for p in pngs if os.path.exists(p)]
    if not imgs:
        return None
    cw = (width - 80 * (cols + 1)) // cols
    rows = [imgs[i:i + cols] for i in range(0, len(imgs), cols)]
    heights = [max(int(im.height * cw / im.width) for im in r) for r in rows]
    H = 180 + sum(heights) + 80 * len(rows)
    c = Image.new("RGBA", (width, H), PAGE)
    ImageDraw.Draw(c).text((80, 60), title, font=_font(56, True), fill=INK)
    y = 180
    for r, h in zip(rows, heights):
        x = 80
        for im in r:
            hh = int(im.height * cw / im.width)
            _paste_card(c, im, (x, y, x + cw, y + hh), radius=18, contain=True)
            x += cw + 80
        y += h + 80
    c.convert("RGB").save(out_path, quality=92)
    return out_path
