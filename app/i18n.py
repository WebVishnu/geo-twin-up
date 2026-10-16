"""English / Hindi UI strings. Keys are identical in both languages (checked by a test)."""

from __future__ import annotations

LANGS = {"en": "English", "hi": "हिन्दी"}

STR: dict[str, dict[str, str]] = {
    "en": {
        "app_title": "Geo Twin AI · Uttar Pradesh",
        "tagline": "Rainfall for all 75 districts, a 7-day outlook, and a what-if tool. A student prototype, updated once a day.",
        "language": "Language",
        "nav_map": "Map",
        "nav_whatif": "What if",
        "nav_method": "Method and limits",
        "layer": "Map layer",
        "layer_obs": "Observed rainfall (IMD)",
        "layer_fc": "Forecast rain, next 7 days",
        "period": "Period",
        "period_days": "Last {n} days of data",
        "period_monsoon": "Last monsoon (June to September {y})",
        "no_rain_period": "No rain was recorded in this period, so every district shows 0 mm. Pick a longer period or the monsoon.",
        "rain_mm": "Rain (mm)",
        "forecast_mm": "Forecast rain (mm)",
        "total": "Total",
        "state_avg": "UP average",
        "rank": "Rank {r} of {n} (1 is wettest)",
        "district": "District",
        "pick_district": "Pick a district, or click it on the map",
        "no_district": "Click a district on the map to see its details.",
        "daily_rain": "Daily rain",
        "temp_range": "Forecast temperature range: {lo:.0f} to {hi:.0f} °C",
        "data_through": "IMD data through {d}",
        "forecast_issued": "Forecast issued {d}",
        "no_forecast": "No forecast has been logged yet. Run the daily logger, or wait for the scheduled job.",
        "no_obs": "No rainfall data found. Run scripts/build_district_data.py first.",
        "hover_hint": "Hover for values · click to select",
        "whatif_title": "What if the monsoon changes?",
        "whatif_intro": "Change rainfall or temperature and see how past monsoons would have looked with that change.",
        "banner": "Sensitivity analysis, not a forecast. It re-scores past years with your change. It does not predict the future.",
        "area": "Area",
        "all_up": "Whole of Uttar Pradesh (average)",
        "rain_change": "Rain change (%)",
        "heavy_change": "Extra change on heavy-rain days (%)",
        "temp_change": "Temperature change (°C)",
        "temp_missing": "Temperature data is not loaded yet, so this slider has no effect.",
        "temp_source": "Temperature is ERA5-Land reanalysis at each district's centre point. It is not an IMD observation.",
        "presets": "Quick scenarios",
        "preset_dry20": "20% less rain",
        "preset_dry30": "30% less rain",
        "preset_reset": "Reset",
        "need_years": "The what-if tool needs at least {n} years of rainfall data. Found {have}.",
        "demo_btn": "Preview with synthetic demo data",
        "demo_banner": "DEMO DATA. These numbers are synthetic and describe no real place.",
        "drought_share": "Years below 75% of normal monsoon rain",
        "range_to": "to",
        "before": "Before",
        "after": "With your change",
        "change": "Change",
        "ci_note": "The small range under each change is the 90% bootstrap band over years.",
        "heavy_note": "Heavy-rain days (64.5 mm or more) are hidden here because averaging 75 districts smooths them out. Pick one district to see them.",
        "chart_ratio": "Monsoon rain as % of normal, by year",
        "line75": "75% of normal: IMD drought line",
        "year": "Year",
        "ind_jjas_ratio_to_lpa": "Monsoon rain (share of normal)",
        "ind_spi3_monsoon_mean": "Drought index SPI-3 (monsoon)",
        "ind_rx1day_mm": "Wettest day of year (mm)",
        "ind_rx5day_mm": "Wettest 5 days of year (mm)",
        "ind_heavy_rain_days": "Heavy-rain days per year (64.5 mm or more)",
        "ind_heatwave_days": "Heat-wave days per year",
        "method_title": "How this works and where it falls short",
        "method_body": """
**What you are looking at.** A digital-twin-style decision-support prototype. It takes observed rainfall, scores it for each district, and lets you test simple changes. It is not a live twin and it does not feed anything back into the real world.

**Data.**
- Rainfall: IMD gridded daily rain at 0.25° (about 25 km), averaged over district areas.
- Forecast: Open-Meteo, logged once a day for a grid over Uttar Pradesh.
- Districts: geoBoundaries (ODbL), 75 districts, simplified for speed.
- Temperature maps are not shown yet. IMD's temperature archive is 1° (about 100 km), too coarse for district maps.

**What-if.** We scale observed daily rain and shift observed temperature, then recompute indicators against fixed baseline statistics. It is a sensitivity test. It never runs the forecast models, because a world with 30% less rain is outside what they learned.

**Known limits.**
- In our 2024 to 2025 tests, a model that uses past rainfall alone beat climatology only for the next day. Adding weather-model forecasts helped clearly at 1 day, only slightly at 3 days, and not at all by 7 days. The forecast layer on the map shows the weather model's own district averages, uncorrected, and is least reliable for heavy rain.
- District values are averages over grid cells. They can miss a local storm.
- Heat-wave and heavy-rain thresholds follow IMD conventions as we understood them. Check them against IMD before relying on them.
- This is not an official warning service. For warnings, use IMD.

**Related work.** Mission Mausam, IMD's Bharat Forecast System and its AI forecast pilot for Uttar Pradesh, Destination Earth, and GAEA for Cyprus work on similar problems at larger scale. To our knowledge, based on public sources we could find, we did not find an openly accessible, district-level tool for Uttar Pradesh that combines open forecast data, hazard indices and interactive what-if analysis in one place.
""",
        "credits": "Data: IMD, Open-Meteo, geoBoundaries (ODbL). Prototype by students of ABES Institute of Technology.",
    },
    "hi": {
        "app_title": "जियो ट्विन AI · उत्तर प्रदेश",
        "tagline": "सभी 75 जिलों की वर्षा, 7 दिन का अनुमान और 'अगर ऐसा हो तो' टूल। यह छात्रों का प्रोटोटाइप है, दिन में एक बार अपडेट होता है।",
        "language": "भाषा",
        "nav_map": "नक्शा",
        "nav_whatif": "अगर ऐसा हो तो",
        "nav_method": "तरीका और सीमाएँ",
        "layer": "नक्शे की परत",
        "layer_obs": "दर्ज की गई वर्षा (IMD)",
        "layer_fc": "अनुमानित वर्षा, अगले 7 दिन",
        "period": "अवधि",
        "period_days": "डेटा के आख़िरी {n} दिन",
        "period_monsoon": "पिछला मानसून (जून से सितंबर {y})",
        "no_rain_period": "इस अवधि में कोई वर्षा दर्ज नहीं हुई, इसलिए हर जिला 0 मिमी दिखाता है। लंबी अवधि या मानसून चुनें।",
        "rain_mm": "वर्षा (मिमी)",
        "forecast_mm": "अनुमानित वर्षा (मिमी)",
        "total": "कुल",
        "state_avg": "UP का औसत",
        "rank": "{n} में से {r}वाँ स्थान (1 सबसे ज़्यादा बारिश)",
        "district": "जिला",
        "pick_district": "जिला चुनें, या नक्शे पर क्लिक करें",
        "no_district": "विवरण देखने के लिए नक्शे पर किसी जिले पर क्लिक करें।",
        "daily_rain": "रोज़ की वर्षा",
        "temp_range": "अनुमानित तापमान: {lo:.0f} से {hi:.0f} °C",
        "data_through": "IMD डेटा {d} तक",
        "forecast_issued": "अनुमान जारी: {d}",
        "no_forecast": "अभी कोई अनुमान दर्ज नहीं हुआ है। रोज़ वाला लॉगर चलाएँ या तय समय का इंतज़ार करें।",
        "no_obs": "वर्षा का डेटा नहीं मिला। पहले scripts/build_district_data.py चलाएँ।",
        "hover_hint": "मान देखने के लिए ऊपर ले जाएँ · चुनने के लिए क्लिक करें",
        "whatif_title": "अगर मानसून बदल जाए तो?",
        "whatif_intro": "वर्षा या तापमान बदलें और देखें कि उस बदलाव के साथ बीते मानसून कैसे दिखते।",
        "banner": "यह संवेदनशीलता विश्लेषण है, पूर्वानुमान नहीं। यह बीते वर्षों को आपके बदलाव के साथ दोबारा आँकता है, भविष्य नहीं बताता।",
        "area": "क्षेत्र",
        "all_up": "पूरा उत्तर प्रदेश (औसत)",
        "rain_change": "वर्षा में बदलाव (%)",
        "heavy_change": "भारी बारिश वाले दिनों में अतिरिक्त बदलाव (%)",
        "temp_change": "तापमान में बदलाव (°C)",
        "temp_missing": "तापमान का डेटा अभी लोड नहीं है, इसलिए इस स्लाइडर का असर नहीं दिखेगा।",
        "temp_source": "तापमान ERA5-Land रीएनालिसिस है, जो हर जिले के केंद्र बिंदु पर लिया गया है। यह IMD का अवलोकन नहीं है।",
        "presets": "तुरंत आज़माएँ",
        "preset_dry20": "20% कम बारिश",
        "preset_dry30": "30% कम बारिश",
        "preset_reset": "रीसेट",
        "need_years": "इस टूल के लिए कम से कम {n} साल का वर्षा डेटा चाहिए। अभी {have} साल मिले।",
        "demo_btn": "नकली डेमो डेटा से झलक देखें",
        "demo_banner": "डेमो डेटा। ये आँकड़े नकली हैं और किसी असली जगह के नहीं हैं।",
        "drought_share": "वे वर्ष जब मानसून की वर्षा सामान्य के 75% से कम रही",
        "range_to": "से",
        "before": "पहले",
        "after": "आपके बदलाव के साथ",
        "change": "अंतर",
        "ci_note": "हर अंतर के नीचे छोटा दायरा वर्षों पर 90% बूटस्ट्रैप सीमा है।",
        "heavy_note": "भारी वर्षा के दिन (64.5 मिमी या अधिक) यहाँ छिपे हैं, क्योंकि 75 जिलों का औसत उन्हें समतल कर देता है। देखने के लिए एक जिला चुनें।",
        "chart_ratio": "हर साल मानसून की वर्षा, सामान्य का %",
        "line75": "सामान्य का 75%: IMD की सूखे की रेखा",
        "year": "वर्ष",
        "ind_jjas_ratio_to_lpa": "मानसून की वर्षा (सामान्य का अनुपात)",
        "ind_spi3_monsoon_mean": "सूखा सूचकांक SPI-3 (मानसून)",
        "ind_rx1day_mm": "साल का सबसे भीगा दिन (मिमी)",
        "ind_rx5day_mm": "साल के सबसे भीगे 5 दिन (मिमी)",
        "ind_heavy_rain_days": "प्रति वर्ष भारी वर्षा के दिन (64.5 मिमी या अधिक)",
        "ind_heatwave_days": "प्रति वर्ष लू के दिन",
        "method_title": "यह कैसे काम करता है और कहाँ कमज़ोर है",
        "method_body": """
**आप क्या देख रहे हैं।** यह डिजिटल-ट्विन जैसा निर्णय-सहायता प्रोटोटाइप है। यह दर्ज की गई वर्षा को हर जिले के लिए आँकता है और सरल बदलाव आज़माने देता है। यह लाइव ट्विन नहीं है और असली दुनिया में कुछ वापस नहीं भेजता।

**डेटा।**
- वर्षा: IMD की 0.25° (लगभग 25 किमी) की रोज़ की ग्रिड वर्षा, जिलों के क्षेत्र पर औसत की गई।
- अनुमान: Open-Meteo, उत्तर प्रदेश की ग्रिड के लिए दिन में एक बार दर्ज।
- जिले: geoBoundaries (ODbL), 75 जिले, तेज़ी के लिए सरल किए गए।
- तापमान के नक्शे अभी नहीं हैं। IMD का तापमान संग्रह 1° (लगभग 100 किमी) का है, जो जिला नक्शों के लिए बहुत मोटा है।

**अगर ऐसा हो तो।** हम दर्ज की गई रोज़ की वर्षा को घटाते-बढ़ाते हैं और तापमान खिसकाते हैं, फिर तय आधार आँकड़ों के सामने सूचकांक दोबारा निकालते हैं। यह संवेदनशीलता जाँच है। यह अनुमान वाले मॉडल नहीं चलाती, क्योंकि 30% कम बारिश वाली दुनिया उनके सीखे हुए दायरे से बाहर है।

**ज्ञात सीमाएँ।**
- हमारे 2024 से 2025 के परीक्षण में, सिर्फ़ पिछली वर्षा पर बना मॉडल केवल अगले दिन के लिए जलवायु-औसत से बेहतर रहा। मौसम मॉडल के अनुमान जोड़ने पर 1 दिन में साफ़ फ़ायदा हुआ, 3 दिन में बहुत थोड़ा, और 7 दिन तक कुछ नहीं। नक्शे की अनुमान वाली परत मौसम मॉडल के जिला-औसत मान बिना सुधार के दिखाती है, और भारी वर्षा के लिए सबसे कम भरोसेमंद है।
- जिले के मान ग्रिड के औसत हैं। स्थानीय तूफ़ान छूट सकता है।
- लू और भारी वर्षा की सीमाएँ IMD के नियमों के हमारे समझे रूप पर हैं। भरोसा करने से पहले IMD से मिलाएँ।
- यह आधिकारिक चेतावनी सेवा नहीं है। चेतावनी के लिए IMD देखें।

**संबंधित काम।** मिशन मौसम, IMD का भारत फोरकास्ट सिस्टम और उत्तर प्रदेश के लिए उसका AI पायलट, डेस्टिनेशन अर्थ और साइप्रस का GAEA बड़े पैमाने पर मिलती-जुलती समस्याओं पर काम करते हैं। हमारी जानकारी में, जो सार्वजनिक स्रोत हमें मिले उनके आधार पर, हमें उत्तर प्रदेश के लिए ऐसा खुला, जिला-स्तर का टूल नहीं मिला जो खुला अनुमान डेटा, जोखिम सूचकांक और संवादात्मक 'अगर ऐसा हो तो' विश्लेषण एक जगह दे।
""",
        "credits": "डेटा: IMD, Open-Meteo, geoBoundaries (ODbL)। ABES इंस्टीट्यूट ऑफ टेक्नोलॉजी के छात्रों का प्रोटोटाइप।",
    },
}


def make_t(lang: str):
    """Return a translator bound to a language; unknown keys raise KeyError loudly."""
    table = STR[lang]

    def t(key: str, **kw: object) -> str:
        s = table[key]
        return s.format(**kw) if kw else s

    return t
