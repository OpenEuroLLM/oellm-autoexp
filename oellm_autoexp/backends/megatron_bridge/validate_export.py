"""Load an HF-exported checkpoint and run canonical-prompt generations as a
post-conversion sanity check.

Writes a JSON report (one completion per prompt) so a Megatron -> HF
conversion can be spot-checked for "does this load and produce coherent,
plausible text" across all OpenEuroLLM target languages, not just
English.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 4 canonical incomplete-sentence prompts, translated into each of the 36
# OpenEuroLLM target languages (EU official + co-official + candidate-member +
# closely associated Scandinavian). Translations were produced with a bias
# toward native fluency/naturalness rather than literal word-for-word
# translation (e.g. prompt_4 uses each language's actual fairy-tale-opening
# convention). "confidence" is a self-reported translation-quality flag;
# gle/eus/mlt/kat/lav/lit/sqi/isl are the ones most worth a native-speaker
# spot-check.
CANONICAL_PROMPTS = {
    "bul": {
        "language_name": "Bulgarian",
        "prompt_1": "Столицата на Франция е",
        "prompt_2": "Обратното на горещо е",
        "prompt_3": "Водата кипи при температура от",
        "prompt_4": "Имало едно време,",
        "confidence": "high",
    },
    "ces": {
        "language_name": "Czech",
        "prompt_1": "Hlavním městem Francie je",
        "prompt_2": "Opakem horka je",
        "prompt_3": "Voda vře při teplotě",
        "prompt_4": "Bylo, nebylo,",
        "confidence": "high",
    },
    "dan": {
        "language_name": "Danish",
        "prompt_1": "Frankrigs hovedstad er",
        "prompt_2": "Det modsatte af varm er",
        "prompt_3": "Vand koger ved en temperatur på",
        "prompt_4": "Der var engang,",
        "confidence": "high",
    },
    "deu": {
        "language_name": "German",
        "prompt_1": "Die Hauptstadt von Frankreich ist",
        "prompt_2": "Das Gegenteil von heiß ist",
        "prompt_3": "Wasser kocht bei einer Temperatur von",
        "prompt_4": "Es war einmal,",
        "confidence": "high",
    },
    "ell": {
        "language_name": "Greek",
        "prompt_1": "Η πρωτεύουσα της Γαλλίας είναι",
        "prompt_2": "Το αντίθετο του ζεστού είναι",
        "prompt_3": "Το νερό βράζει σε θερμοκρασία",
        "prompt_4": "Μια φορά κι έναν καιρό,",
        "confidence": "high",
    },
    "eng": {
        "language_name": "English",
        "prompt_1": "The capital of France is",
        "prompt_2": "The opposite of hot is",
        "prompt_3": "Water boils at a temperature of",
        "prompt_4": "Once upon a time,",
        "confidence": "high",
    },
    "est": {
        "language_name": "Estonian",
        "prompt_1": "Prantsusmaa pealinn on",
        "prompt_2": "Kuuma vastand on",
        "prompt_3": "Vesi keeb temperatuuril",
        "prompt_4": "Elas kord,",
        "confidence": "high",
    },
    "fin": {
        "language_name": "Finnish",
        "prompt_1": "Ranskan pääkaupunki on",
        "prompt_2": "Kuuman vastakohta on",
        "prompt_3": "Vesi kiehuu lämpötilassa",
        "prompt_4": "Olipa kerran,",
        "confidence": "high",
    },
    "fra": {
        "language_name": "French",
        "prompt_1": "La capitale de la France est",
        "prompt_2": "Le contraire de chaud est",
        "prompt_3": "L'eau bout à une température de",
        "prompt_4": "Il était une fois,",
        "confidence": "high",
    },
    "gle": {
        "language_name": "Irish",
        "prompt_1": "Is é príomhchathair na Fraince ná",
        "prompt_2": "Is é a mhalairt de theas ná",
        "prompt_3": "Fiuchann uisce ag teocht",
        "prompt_4": "Fadó, fadó,",
        "confidence": "low",
    },
    "hrv": {
        "language_name": "Croatian",
        "prompt_1": "Glavni grad Francuske je",
        "prompt_2": "Suprotno od vruće je",
        "prompt_3": "Voda ključa na temperaturi od",
        "prompt_4": "Bio jednom jedan,",
        "confidence": "high",
    },
    "hun": {
        "language_name": "Hungarian",
        "prompt_1": "Franciaország fővárosa",
        "prompt_2": "A meleg ellentéte",
        "prompt_3": "A víz forráspontja",
        "prompt_4": "Egyszer volt, hol nem volt,",
        "confidence": "high",
    },
    "ita": {
        "language_name": "Italian",
        "prompt_1": "La capitale della Francia è",
        "prompt_2": "Il contrario di caldo è",
        "prompt_3": "L'acqua bolle a una temperatura di",
        "prompt_4": "C'era una volta,",
        "confidence": "high",
    },
    "lav": {
        "language_name": "Latvian",
        "prompt_1": "Francijas galvaspilsēta ir",
        "prompt_2": "Karsta pretstats ir",
        "prompt_3": "Ūdens vārās temperatūrā",
        "prompt_4": "Reiz sen senos laikos,",
        "confidence": "medium",
    },
    "lit": {
        "language_name": "Lithuanian",
        "prompt_1": "Prancūzijos sostinė yra",
        "prompt_2": "Karšto priešingybė yra",
        "prompt_3": "Vandens virimo temperatūra yra",
        "prompt_4": "Kartą, seniai seniai,",
        "confidence": "medium",
    },
    "mlt": {
        "language_name": "Maltese",
        "prompt_1": "Il-belt kapitali ta' Franza hija",
        "prompt_2": "Il-kontra ta' sħun hija",
        "prompt_3": "L-ilma jagħli f'temperatura ta'",
        "prompt_4": "Darba waħda,",
        "confidence": "low",
    },
    "nld": {
        "language_name": "Dutch",
        "prompt_1": "De hoofdstad van Frankrijk is",
        "prompt_2": "Het tegenovergestelde van warm is",
        "prompt_3": "Water kookt bij een temperatuur van",
        "prompt_4": "Er was eens,",
        "confidence": "high",
    },
    "pol": {
        "language_name": "Polish",
        "prompt_1": "Stolicą Francji jest",
        "prompt_2": "Przeciwieństwem gorąca jest",
        "prompt_3": "Woda wrze w temperaturze",
        "prompt_4": "Dawno, dawno temu,",
        "confidence": "high",
    },
    "por": {
        "language_name": "Portuguese",
        "prompt_1": "A capital da França é",
        "prompt_2": "O contrário de quente é",
        "prompt_3": "A água ferve a uma temperatura de",
        "prompt_4": "Era uma vez,",
        "confidence": "high",
    },
    "ron": {
        "language_name": "Romanian",
        "prompt_1": "Capitala Franței este",
        "prompt_2": "Opusul lui cald este",
        "prompt_3": "Apa fierbe la o temperatură de",
        "prompt_4": "A fost odată ca niciodată,",
        "confidence": "high",
    },
    "slk": {
        "language_name": "Slovak",
        "prompt_1": "Hlavným mestom Francúzska je",
        "prompt_2": "Opakom horúceho je",
        "prompt_3": "Voda vrie pri teplote",
        "prompt_4": "Kde bolo, tam bolo,",
        "confidence": "high",
    },
    "slv": {
        "language_name": "Slovenian",
        "prompt_1": "Glavno mesto Francije je",
        "prompt_2": "Nasprotje vročega je",
        "prompt_3": "Voda zavre pri temperaturi",
        "prompt_4": "Nekoč je živel,",
        "confidence": "high",
    },
    "spa": {
        "language_name": "Spanish",
        "prompt_1": "La capital de Francia es",
        "prompt_2": "Lo contrario de caliente es",
        "prompt_3": "El agua hierve a una temperatura de",
        "prompt_4": "Érase una vez,",
        "confidence": "high",
    },
    "swe": {
        "language_name": "Swedish",
        "prompt_1": "Frankrikes huvudstad är",
        "prompt_2": "Motsatsen till varm är",
        "prompt_3": "Vatten kokar vid en temperatur på",
        "prompt_4": "Det var en gång,",
        "confidence": "high",
    },
    "cat": {
        "language_name": "Catalan",
        "prompt_1": "La capital de França és",
        "prompt_2": "El contrari de calent és",
        "prompt_3": "L'aigua bull a una temperatura de",
        "prompt_4": "Hi havia una vegada,",
        "confidence": "high",
    },
    "eus": {
        "language_name": "Basque",
        "prompt_1": "Frantziako hiriburua",
        "prompt_2": "Beroaren kontrakoa",
        "prompt_3": "Uraren irakite-puntua",
        "prompt_4": "Bazen behin,",
        "confidence": "low",
    },
    "glg": {
        "language_name": "Galician",
        "prompt_1": "A capital de Francia é",
        "prompt_2": "O contrario de quente é",
        "prompt_3": "A auga ferve a unha temperatura de",
        "prompt_4": "Había unha vez,",
        "confidence": "high",
    },
    "bos": {
        "language_name": "Bosnian",
        "prompt_1": "Glavni grad Francuske je",
        "prompt_2": "Suprotno od vrućeg je",
        "prompt_3": "Voda ključa na temperaturi od",
        "prompt_4": "Bio jednom jedan,",
        "confidence": "high",
    },
    "kat": {
        "language_name": "Georgian",
        "prompt_1": "საფრანგეთის დედაქალაქია",
        "prompt_2": "ცხელის საპირისპიროა",
        "prompt_3": "წყალი დუღს ტემპერატურაზე",
        "prompt_4": "იყო და არა იყო რა,",
        "confidence": "medium",
    },
    "mkd": {
        "language_name": "Macedonian",
        "prompt_1": "Главниот град на Франција е",
        "prompt_2": "Спротивното на топло е",
        "prompt_3": "Водата врие на температура од",
        "prompt_4": "Си беше еднаш,",
        "confidence": "high",
    },
    "sqi": {
        "language_name": "Albanian",
        "prompt_1": "Kryeqyteti i Francës është",
        "prompt_2": "E kundërta e nxehtit është",
        "prompt_3": "Uji vlon në temperaturën prej",
        "prompt_4": "Na ishte një herë,",
        "confidence": "medium",
    },
    "srp": {
        "language_name": "Serbian",
        "prompt_1": "Главни град Француске је",
        "prompt_2": "Супротно од вруће је",
        "prompt_3": "Вода кључа на температури од",
        "prompt_4": "Био једном један,",
        "confidence": "high",
    },
    "tur": {
        "language_name": "Turkish",
        "prompt_1": "Fransa'nın başkenti",
        "prompt_2": "Sıcağın zıttı",
        "prompt_3": "Suyun kaynama sıcaklığı",
        "prompt_4": "Bir varmış, bir yokmuş,",
        "confidence": "high",
    },
    "ukr": {
        "language_name": "Ukrainian",
        "prompt_1": "Столицею Франції є",
        "prompt_2": "Протилежне до гарячого —",
        "prompt_3": "Вода кипить за температури",
        "prompt_4": "Жили-були,",
        "confidence": "high",
    },
    "isl": {
        "language_name": "Icelandic",
        "prompt_1": "Höfuðborg Frakklands er",
        "prompt_2": "Andstæðan við heitt er",
        "prompt_3": "Vatn sýður við hitastig upp á",
        "prompt_4": "Einu sinni var,",
        "confidence": "medium",
    },
    "nor": {
        "language_name": "Norwegian",
        "prompt_1": "Frankrikes hovedstad er",
        "prompt_2": "Det motsatte av varm er",
        "prompt_3": "Vann koker ved en temperatur på",
        "prompt_4": "Det var en gang,",
        "confidence": "high",
    },
}


def validate(hf_path: Path, output_json: Path, max_new_tokens: int = 30) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(hf_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(hf_path), dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    model.eval()

    results = []
    for lang_code, entry in CANONICAL_PROMPTS.items():
        for i in range(1, 5):
            prompt = entry[f"prompt_{i}"]
            inputs = tok(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id
                    if tok.pad_token_id is not None
                    else tok.eos_token_id,
                )
            text = tok.decode(out[0], skip_special_tokens=True)
            results.append(
                {
                    "lang": lang_code,
                    "language_name": entry["language_name"],
                    "prompt_index": i,
                    "prompt": prompt,
                    "generation": text,
                }
            )

    report = {"hf_path": str(hf_path), "status": "ok", "prompts": results}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-path", required=True, type=Path)
    ap.add_argument("--output-json", required=True, type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=30)
    return ap.parse_args()


def main() -> int:
    args = _parse()
    try:
        validate(args.hf_path, args.output_json, args.max_new_tokens)
    except Exception as exc:  # noqa: BLE001
        report = {"hf_path": str(args.hf_path), "status": "error", "error": str(exc)}
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
