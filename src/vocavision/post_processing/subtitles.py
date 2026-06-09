"""ASS subtitle generation using pysubs2."""

from __future__ import annotations

from pathlib import Path

from vocavision.utils.text_utils import highlight_target_words, mask_target_words


class SubtitleRenderer:
    def render(
        self,
        *,
        text: str,
        target_words: list[str],
        duration_sec: float,
        output_path: Path,
        masked_words: list[str] | None = None,
    ) -> Path:
        rendered_text = highlight_target_words(text, target_words)
        if masked_words:
            rendered_text = mask_target_words(
                rendered_text,
                masked_words,
                replacement="_____",
                highlight_mask=True,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pysubs2

            subs = pysubs2.SSAFile()
            style = pysubs2.SSAStyle()
            style.fontname = "Arial"
            style.fontsize = 20
            style.primarycolor = pysubs2.Color(255, 255, 255, 0)
            style.outlinecolor = pysubs2.Color(0, 0, 0, 0)
            style.backcolor = pysubs2.Color(0, 0, 0, 0)
            style.bold = True
            style.alignment = pysubs2.Alignment.BOTTOM_CENTER
            style.marginv = 24
            style.outline = 1.5
            style.shadow = 0.5
            subs.styles["Default"] = style
            end_ms = max(1, int(duration_sec * 1000))
            subs.events.append(pysubs2.SSAEvent(start=0, end=end_ms, text=rendered_text, style="Default"))
            subs.save(str(output_path))
        except ModuleNotFoundError:
            end_cs = max(1, int(duration_sec * 100))
            ass_text = (
                "[Script Info]\n"
                "ScriptType: v4.00+\n"
                "\n"
                "[V4+ Styles]\n"
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour,"
                " Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline,"
                " Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
                "Style: Default,Arial,20,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,1.5,"
                "0.5,2,20,20,24,1\n"
                "\n"
                "[Events]\n"
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                f"Dialogue: 0,0:00:00.00,0:00:{end_cs // 100:02d}.{end_cs % 100:02d},Default,,0,0,0,,{rendered_text}\n"
            )
            output_path.write_text(ass_text, encoding="utf-8")
        return output_path
