"""
Records the intro film (docs/intro.html) frame by frame into assets/intro.gif, for the top of the README.
GitHub READMEs cannot run JavaScript, but they do play GIFs, so this is how the film reaches github.com.

    pip install playwright && python -m playwright install chromium     # once
    sudo apt install -y ffmpeg                                           # once
    python tools/record_intro.py            # about a minute

Because the film is a pure function of time, every frame is exact: no screen-recording jitter.
"""
import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
FPS, WIDTH = 10, 960


async def main() -> None:
    frames = Path(tempfile.mkdtemp())
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        await page.goto((ROOT / "docs" / "intro.html").as_uri() + "?record")
        await page.wait_for_function("window.INTRO !== undefined")
        duration = await page.evaluate("INTRO.duration")
        n = int(duration * FPS)
        for i in range(n):
            await page.evaluate(f"INTRO.seek({i / FPS})")
            await page.locator("canvas").screenshot(path=str(frames / f"f{i:04d}.png"))
            if i % 50 == 0:
                print(f"  frame {i}/{n}")
        await browser.close()
    out = ROOT / "assets" / "intro.gif"
    pal = frames / "palette.png"
    scale = f"fps={FPS},scale={WIDTH}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(frames / "f%04d.png"), "-vf", f"{scale},palettegen=max_colors=96:stats_mode=diff", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "f%04d.png"), "-i", str(pal),
                    "-lavfi", f"{scale} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle", "-loop", "0", str(out)], check=True)
    shutil.rmtree(frames)
    print(f"wrote {out.relative_to(ROOT)}  ({out.stat().st_size / 1e6:.1f} MB, {n} frames)")


if __name__ == "__main__":
    asyncio.run(main())
