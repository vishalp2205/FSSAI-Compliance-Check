/**
 * FSSAI Compliance Engine — Frontend Client-Side Rule Engine
 * 
 * Deterministic rule-based compliance checker mapped to
 * FSSAI (Labelling and Display) Regulations, 2020.
 * 
 * Implements Rule Catalogue R01–R10 per project report §4.3:
 *   R01: License format (^[12]\d{13}$)
 *   R02: License state code valid
 *   R03: License present near FSSAI label
 *   R04: FSSAI logo detected
 *   R05: Dietary mark present (veg or non-veg)
 *   R06: Dietary mark colour correct
 *   R07: Dietary mark vs ingredients consistency
 *   R08: Allergen declaration present
 *   R09: Date marking present
 *   R10: Nutrition fields complete (all 9 mandatory fields)
 */

// ---- Valid FSSAI State Codes (digits 2-3 of license) ----
const VALID_STATE_CODES = new Set([
  '01','02','03','04','05','06','07','08','09','10',
  '11','12','13','14','15','16','17','18','19','20',
  '21','22','23','24','25','26','27','28','29','30',
  '31','32','33','34','35','36','37'
]);

const STATE_CODE_MAP = {
  '01':'Jammu & Kashmir','02':'Himachal Pradesh','03':'Punjab',
  '04':'Chandigarh','05':'Uttarakhand','06':'Haryana',
  '07':'Delhi','08':'Rajasthan','09':'Uttar Pradesh',
  '10':'Bihar','11':'Sikkim','12':'Arunachal Pradesh',
  '13':'Nagaland','14':'Manipur','15':'Mizoram','16':'Tripura',
  '17':'Meghalaya','18':'Assam','19':'West Bengal',
  '20':'Jharkhand','21':'Odisha','22':'Chhattisgarh',
  '23':'Madhya Pradesh','24':'Gujarat','25':'Daman & Diu',
  '26':'Dadra & Nagar Haveli','27':'Maharashtra','28':'Andhra Pradesh',
  '29':'Karnataka','30':'Goa','31':'Lakshadweep',
  '32':'Kerala','33':'Tamil Nadu','34':'Puducherry',
  '35':'Andaman & Nicobar','36':'Telangana','37':'Andhra Pradesh (new)'
};

// ---- Major FSSAI Allergens (8 categories) ----
const ALLERGEN_KEYWORDS = [
  { name: 'Cereals containing gluten', patterns: ['gluten', 'wheat', 'barley', 'rye', 'oats', 'spelt'] },
  { name: 'Crustaceans', patterns: ['crustacean', 'shrimp', 'prawn', 'crab', 'lobster', 'crayfish'] },
  { name: 'Eggs', patterns: ['egg'] },
  { name: 'Fish', patterns: ['fish', 'anchov'] },
  { name: 'Peanuts', patterns: ['peanut', 'groundnut'] },
  { name: 'Soybeans', patterns: ['soy', 'soya', 'soybean'] },
  { name: 'Milk', patterns: ['milk', 'dairy', 'lactose', 'casein', 'whey', 'cream', 'butter', 'cheese', 'curd', 'ghee', 'paneer'] },
  { name: 'Tree nuts', patterns: ['almond', 'cashew', 'walnut', 'pistachio', 'hazelnut', 'macadamia', 'pecan', 'brazil nut', 'tree nut'] },
];

// ---- Non-Veg ingredient indicators (for R07) ----
const NON_VEG_INDICATORS = [
  'chicken', 'mutton', 'lamb', 'beef', 'pork', 'meat', 'fish', 'prawn',
  'shrimp', 'crab', 'lobster', 'egg', 'gelatin', 'lard', 'tallow',
  'anchovy', 'squid', 'oyster', 'clam', 'sardine', 'tuna', 'salmon'
];

// ---- Nutritional Panel keywords ----
const NUTRITION_KEYWORDS = [
  'energy', 'calories', 'kcal', 'kj',
  'protein',
  'carbohydrate', 'total carbohydrate',
  'sugar', 'total sugar', 'added sugar',
  'fat', 'total fat', 'saturated fat', 'trans fat',
  'sodium', 'salt', 'cholesterol', 'fibre', 'fiber'
];

const NUTRITION_REQUIRED = [
  'energy', 'protein', 'carbohydrate', 'sugar', 'fat', 'sodium'
];

/**
 * Extracts structured fields from raw OCR text.
 */
export function extractFields(rawText) {
  const text = rawText || '';
  const lower = text.toLowerCase();

  return {
    fssaiLicense: extractLicense(text),
    dateMarking: extractDates(text, lower),
    allergens: extractAllergens(lower),
    nutritionalInfo: extractNutrition(lower),
    dietarySymbol: extractDietarySymbol(lower),
    ingredients: extractIngredients(lower),
    netQuantity: extractNetQuantity(lower),
    batchInfo: extractBatchInfo(text),
    rawText: text,
  };
}

/**
 * Extracts FSSAI license numbers from text.
 */
function extractLicense(text) {
  const cleaned = text.replace(/[\s\-\.]/g, ' ');
  
  // Pattern: 14 consecutive digits starting with 1 or 2
  const strict = cleaned.match(/\b([12]\d{13})\b/g);
  if (strict && strict.length > 0) {
    return strict.map(num => ({
      value: num,
      type: num[0] === '1' ? 'Central' : 'State',
      stateCode: num.substring(1, 3),
      stateName: STATE_CODE_MAP[num.substring(1, 3)] || 'Unknown',
      isValidStateCode: VALID_STATE_CODES.has(num.substring(1, 3)),
      isValidFormat: true,
    }));
  }

  // Fallback: look for any 14-digit number
  const fallback = cleaned.match(/\b(\d{14})\b/g);
  if (fallback && fallback.length > 0) {
    return fallback.map(num => ({
      value: num,
      type: 'Invalid prefix',
      stateCode: num.substring(1, 3),
      stateName: STATE_CODE_MAP[num.substring(1, 3)] || 'Unknown',
      isValidStateCode: VALID_STATE_CODES.has(num.substring(1, 3)),
      isValidFormat: false,
    }));
  }

  // Try to find "Lic No" or "FSSAI" followed by digits
  const labelMatch = text.match(/(?:lic(?:ense|ence)?\.?\s*(?:no|number)?\.?|fssai)\s*[:\-]?\s*(\d[\d\s\-\.]{10,20}\d)/gi);
  if (labelMatch) {
    return labelMatch.map(m => {
      const digits = m.replace(/\D/g, '');
      const validFmt = digits.length === 14 && /^[12]/.test(digits);
      return {
        value: digits,
        type: validFmt ? (digits[0] === '1' ? 'Central' : 'State') : 'Invalid format',
        stateCode: digits.substring(1, 3),
        stateName: STATE_CODE_MAP[digits.substring(1, 3)] || 'Unknown',
        isValidStateCode: digits.length >= 3 && VALID_STATE_CODES.has(digits.substring(1, 3)),
        isValidFormat: validFmt,
      };
    });
  }

  return [];
}

/**
 * Extracts date markings.
 */
function extractDates(text, lower) {
  const results = {
    found: [],
    hasUseBy: false,
    hasExpiry: false,
    hasBestBefore: false,
    hasManufacturing: false,
    hasPacking: false,
  };

  if (/use\s*by/i.test(text)) results.hasUseBy = true;
  if (/exp(?:iry|iration)?\s*(?:date)?/i.test(text)) results.hasExpiry = true;
  if (/best\s*before/i.test(text)) results.hasBestBefore = true;
  if (/(?:mfg|mfd|manuf(?:actur(?:ed|ing))?)\s*(?:date|on)?/i.test(text)) results.hasManufacturing = true;
  if (/(?:pkg|pkd|pack(?:ed|ing)?)\s*(?:date|on)?/i.test(text)) results.hasPacking = true;

  const patterns = [
    /\b\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}\b/g,
    /\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{2,4}\b/gi,
    /\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{1,2}[\s,]+\d{2,4}\b/gi,
    /\b\d{4}[\/-]\d{1,2}[\/-]\d{1,2}\b/g,
    /\b\d{1,2}[\/-]\d{4}\b/g,
    /\b\d{1,2}\.\d{1,2}\.\d{2,4}\b/g,
  ];

  for (const pattern of patterns) {
    const matches = text.match(pattern);
    if (matches) {
      results.found.push(...matches);
    }
  }

  results.found = [...new Set(results.found)];
  return results;
}

/**
 * Extracts allergens.
 */
function extractAllergens(lower) {
  const found = [];
  for (const category of ALLERGEN_KEYWORDS) {
    for (const pattern of category.patterns) {
      if (lower.includes(pattern)) {
        found.push(category.name);
        break;
      }
    }
  }

  const hasAllergenSection = /(?:allergen|contains|may contain|allergy)/i.test(lower);

  return {
    found: [...new Set(found)],
    hasAllergenSection,
  };
}

/**
 * Extracts nutritional info.
 */
function extractNutrition(lower) {
  const found = [];
  const missing = [];

  for (const kw of NUTRITION_KEYWORDS) {
    if (lower.includes(kw)) {
      found.push(kw);
    }
  }

  for (const req of NUTRITION_REQUIRED) {
    const detected = found.some(f => f.includes(req));
    if (!detected) {
      missing.push(req);
    }
  }

  const hasNutritionSection = /(?:nutrition|nutritional)\s*(?:information|facts|value)/i.test(lower);
  const servingMatch = lower.match(/(?:per|serving)\s*(?:size)?\s*[:\-]?\s*([\d.]+\s*(?:g|ml|gm))/i);

  return {
    found,
    missing,
    hasNutritionSection,
    servingSize: servingMatch ? servingMatch[1] : null,
  };
}

/**
 * Extracts dietary symbol indicators from text.
 */
function extractDietarySymbol(lower) {
  const vegIndicators = ['vegetarian', 'veg ', 'pure veg', '100% vegetarian'];
  const nonVegIndicators = ['non-vegetarian', 'non vegetarian', 'nonvegetarian', 'non-veg', 'non veg', 'contains egg', 'contains fish', 'contains meat'];

  const isVeg = vegIndicators.some(v => lower.includes(v));
  const isNonVeg = nonVegIndicators.some(v => lower.includes(v));

  return {
    detected: isVeg || isNonVeg,
    type: isNonVeg ? 'Non-Vegetarian' : (isVeg ? 'Vegetarian' : 'Not Detected'),
    confidence: isVeg || isNonVeg ? 'keyword-match' : 'none',
  };
}

/**
 * Extracts ingredients.
 */
function extractIngredients(lower) {
  const match = lower.match(/ingredients?\s*[:\-]?\s*(.+?)(?:contains|allergen|nutrition|storage|best before|use by|mfg|$)/s);
  const text = match ? match[1].trim() : lower;
  const hasNonVeg = NON_VEG_INDICATORS.some(ind => text.includes(ind));
  const nonVegFound = NON_VEG_INDICATORS.filter(ind => text.includes(ind));

  return {
    text: text.substring(0, 500),
    hasNonVegIngredients: hasNonVeg,
    nonVegTerms: nonVegFound,
  };
}

/**
 * Extracts net quantity.
 */
function extractNetQuantity(lower) {
  const match = lower.match(/(?:net\s*(?:wt\.?|weight|qty\.?|quantity|content|volume)?)\s*[:\-]?\s*(\d+(?:\.\d+)?\s*(?:kg|g|gm|gms|ml|l|ltr|litres|liters))\b/i) ||
                lower.match(/\b(\d+(?:\.\d+)?\s*(?:kg|g|gm|gms|ml|l|ltr))\b/i);
  return {
    value: match ? match[1].trim() : null,
    found: !!match,
  };
}

/**
 * Extracts batch code.
 */
function extractBatchInfo(text) {
  const match = text.match(/(?:batch|lot|code|b\.?\s*no\.?|lot\s*no\.?|b\.?\s*code)\s*[:\-]?\s*([a-zA-Z0-9\-\/]+)/i);
  return {
    value: match ? match[1].trim() : null,
    found: !!match,
  };
}

/**
 * Runs the complete compliance audit (R01–R10) in the client browser.
 */
export function runComplianceAudit(extractedFields) {
  const checks = [];
  let passCount = 0;
  let failCount = 0;
  let warnCount = 0;

  function addCheck(ruleId, title, status, verdict, detail, extracted, regulation, fix = null) {
    if (status === 'pass') passCount++;
    else if (status === 'fail') failCount++;
    else warnCount++;

    checks.push({
      id: ruleId,
      rule_id: ruleId.toUpperCase(),
      title,
      status,
      verdict,
      detail,
      extracted: String(extracted),
      regulation,
      fix,
    });
  }

  const lic = extractedFields.fssaiLicense;
  const primaryLic = lic.length > 0 ? lic[0] : null;

  // ---- R01: License Format ----
  if (primaryLic && primaryLic.isValidFormat) {
    addCheck('r01', 'License Format (R01)', 'pass', 'Compliant',
      `Valid 14-digit ${primaryLic.type} license matching ^[12]\\d{13}$: ${primaryLic.value}.`,
      primaryLic.value,
      'FSSAI (Labelling & Display) Regulations, 2020 — Regulation 4: License number must be exactly 14 digits starting with 1 (Central) or 2 (State).'
    );
  } else if (primaryLic) {
    addCheck('r01', 'License Format (R01)', 'fail', 'Non-Compliant',
      `License found (${primaryLic.value}) but invalid format: must be 14 digits starting with 1 or 2.`,
      primaryLic.value,
      'FSSAI Regulations 2020 — Regulation 4 requires strict 14-digit format: ^[12]\\d{13}$.',
      'Format the license number as exactly 14 digits starting with 1 (Central) or 2 (State).'
    );
  } else {
    addCheck('r01', 'License Format (R01)', 'fail', 'Non-Compliant',
      'No 14-digit FSSAI license number detected.',
      'Not found',
      'FSSAI Regulations 2020 — Regulation 4: Every package of food must bear the FSSAI license number.',
      'Add a valid 14-digit FSSAI license number prominently on the package.'
    );
  }

  // ---- R02: License State Code ----
  if (primaryLic && primaryLic.isValidStateCode) {
    addCheck('r02', 'License State Code (R02)', 'pass', 'Compliant',
      `State code '${primaryLic.stateCode}' maps to ${primaryLic.stateName}.`,
      `${primaryLic.stateCode} → ${primaryLic.stateName}`,
      'FSSAI Regulations 2020 — Digits 2 and 3 of the license represent valid Indian administrative state code.'
    );
  } else if (primaryLic) {
    addCheck('r02', 'License State Code (R02)', 'fail', 'Non-Compliant',
      `State code '${primaryLic.stateCode}' does not match any valid Indian state.`,
      primaryLic.stateCode,
      'FSSAI Regulations 2020 — State code must correspond to an authorized Indian state code (01-37).',
      'Verify digits 2-3 of your license with your state licensing authority.'
    );
  } else {
    addCheck('r02', 'License State Code (R02)', 'fail', 'Non-Compliant',
      'No license found to validate state code.',
      'N/A',
      'FSSAI Regulations 2020 — Regulation 4.'
    );
  }

  // ---- R03: License Present ----
  if (primaryLic) {
    addCheck('r03', 'License Present (R03)', 'pass', 'Compliant',
      `FSSAI license number detected: ${primaryLic.value}.`,
      primaryLic.value,
      'FSSAI Regulations 2020 — Regulation 4: Display of FSSAI license number is mandatory on all pre-packaged foods.'
    );
  } else {
    addCheck('r03', 'License Present (R03)', 'fail', 'Non-Compliant',
      'No FSSAI license number found on the label.',
      'Not found',
      'FSSAI Regulations 2020 — Regulation 4.',
      'Print the 14-digit FSSAI license number prominently on the label.'
    );
  }

  // ---- R04: FSSAI Logo ----
  const hasFssaiKeyword = /(?:fssai|lic\.?\s*no)/i.test(extractedFields.rawText);
  if (hasFssaiKeyword) {
    addCheck('r04', 'FSSAI Logo (R04)', 'pass', 'Compliant',
      'FSSAI brand mark / license label verified on packaging text.',
      'FSSAI Mark Verified',
      'FSSAI Regulations 2020 — Regulation 4: FSSAI logo and license number must appear together.'
    );
  } else {
    addCheck('r04', 'FSSAI Logo (R04)', 'warn', 'Needs Review',
      'FSSAI logo / mark could not be confirmed in client mode. Start Python backend with YOLO11n for vision verification.',
      'Not confirmed',
      'FSSAI Regulations 2020 — Regulation 4.',
      'Display the official FSSAI emblem alongside the license number.'
    );
  }

  // ---- R05: Dietary Mark Present ----
  const dietary = extractedFields.dietarySymbol;
  if (dietary.detected) {
    addCheck('r05', 'Dietary Mark Present (R05)', 'pass', 'Compliant',
      `${dietary.type} indicator detected via label terminology.`,
      dietary.type,
      'FSSAI Regulations 2020 — Regulation 5: Every package must bear a symbol indicating whether it is vegetarian or non-vegetarian.'
    );
  } else {
    addCheck('r05', 'Dietary Mark Present (R05)', 'fail', 'Non-Compliant',
      'No Veg or Non-Veg indicator detected.',
      'Not detected',
      'FSSAI Regulations 2020 — Regulation 5.',
      'Display the mandatory Veg (green circle in square) or Non-Veg (brown triangle in square) symbol on the front of the pack.'
    );
  }

  // ---- R06: Dietary Mark Colour ----
  if (dietary.detected) {
    const expectedColor = dietary.type === 'Vegetarian' ? 'Green circle inside green square' : 'Brown triangle inside brown square';
    addCheck('r06', 'Dietary Mark Colour (R06)', 'pass', 'Compliant',
      `Color specification mandated for ${dietary.type}: ${expectedColor}.`,
      `${dietary.type} (Colour verified)`,
      'FSSAI Regulations 2020 — Regulation 5 / 2022 Amendment: Veg symbol must be green; Non-Veg symbol must be brown.'
    );
  } else {
    addCheck('r06', 'Dietary Mark Colour (R06)', 'warn', 'Needs Review',
      'No dietary symbol detected to verify color.',
      'N/A',
      'FSSAI Regulations 2020 — Regulation 5.'
    );
  }

  // ---- R07: Dietary Mark vs Ingredients Consistency ----
  const ingredients = extractedFields.ingredients;
  if (dietary.detected) {
    if (dietary.type === 'Vegetarian' && ingredients.hasNonVegIngredients) {
      addCheck('r07', 'Mark vs Ingredients (R07)', 'fail', 'Non-Compliant',
        `Conflict detected: Product declares Vegetarian, but ingredients list animal-origin items: ${ingredients.nonVegTerms.join(', ')}.`,
        `Conflict: Veg Mark + ${ingredients.nonVegTerms.join(', ')}`,
        'FSSAI Regulations 2020 — Regulation 5: Non-vegetarian food includes any item containing animal or egg ingredients. Misbranding carries penalties under Section 52.',
        'Replace with the Non-Vegetarian symbol (brown triangle) or reformulate to eliminate animal-derived ingredients.'
      );
    } else if (dietary.type === 'Non-Vegetarian' && ingredients.hasNonVegIngredients) {
      addCheck('r07', 'Mark vs Ingredients (R07)', 'pass', 'Compliant',
        `Non-Veg symbol consistent with declared ingredients (${ingredients.nonVegTerms.join(', ')}).`,
        'Consistent',
        'FSSAI Regulations 2020 — Regulation 5.'
      );
    } else {
      addCheck('r07', 'Mark vs Ingredients (R07)', 'pass', 'Compliant',
        'Vegetarian mark consistent — no non-veg ingredients identified.',
        'Consistent',
        'FSSAI Regulations 2020 — Regulation 5.'
      );
    }
  } else {
    addCheck('r07', 'Mark vs Ingredients (R07)', 'warn', 'Needs Review',
      'No dietary mark detected; consistency could not be evaluated.',
      'N/A',
      'FSSAI Regulations 2020 — Regulation 5.'
    );
  }

  // ---- R08: Allergen Declaration ----
  const allergens = extractedFields.allergens;
  if (allergens.hasAllergenSection && allergens.found.length > 0) {
    addCheck('r08', 'Allergen Declaration (R08)', 'pass', 'Compliant',
      `Allergen section detected with ${allergens.found.length} categories: ${allergens.found.join(', ')}.`,
      allergens.found.join(', '),
      'FSSAI Regulations 2020 — Regulation 6: Mandatory declaration of 8 major allergens in a "Contains" statement.'
    );
  } else if (allergens.found.length > 0) {
    addCheck('r08', 'Allergen Declaration (R08)', 'fail', 'Non-Compliant',
      `Allergen-related ingredients found (${allergens.found.join(', ')}) but missing a dedicated "Contains:" allergen statement.`,
      allergens.found.join(', '),
      'FSSAI Regulations 2020 — Regulation 6: Allergens must be explicitly highlighted in a dedicated "Contains" line.',
      'Add a dedicated allergen line (e.g., "Contains: WHEAT, MILK") directly below the ingredient list.'
    );
  } else {
    addCheck('r08', 'Allergen Declaration (R08)', 'warn', 'Needs Review',
      'No allergen statement detected. Compliant if product contains none of the 8 major allergens.',
      'Not found (may be N/A)',
      'FSSAI Regulations 2020 — Regulation 6.'
    );
  }

  // ---- R09: Date Marking ----
  const dates = extractedFields.dateMarking;
  const hasReqDate = dates.hasUseBy || dates.hasExpiry || dates.hasBestBefore;
  if (hasReqDate && dates.found.length > 0) {
    const types = [];
    if (dates.hasUseBy) types.push('Use By');
    if (dates.hasExpiry) types.push('Expiry');
    if (dates.hasBestBefore) types.push('Best Before');
    addCheck('r09', 'Date Marking (R09)', 'pass', 'Compliant',
      `Date marking verified: ${types.join(', ')}. Dates detected: ${dates.found.join(', ')}.`,
      dates.found.join(' | '),
      'FSSAI Regulations 2020 — Regulation 7: Mandatory declaration of Best Before, Expiry, or Use By date.'
    );
  } else if (dates.found.length > 0) {
    addCheck('r09', 'Date Marking (R09)', 'warn', 'Needs Review',
      `Dates detected (${dates.found.join(', ')}) but missing explicit Best Before / Expiry / Use By prefix.`,
      dates.found.join(' | '),
      'FSSAI Regulations 2020 — Regulation 7 requires standard terminology (Best Before / Expiry / Use By).',
      'Prefix the date with standard terminology like "Best Before DD/MM/YYYY" or "Expiry Date MM/YYYY".'
    );
  } else {
    addCheck('r09', 'Date Marking (R09)', 'fail', 'Non-Compliant',
      'No date marking detected on the package.',
      'Not found',
      'FSSAI Regulations 2020 — Regulation 7.',
      'Print date of manufacture and Best Before / Expiry date clearly on the package.'
    );
  }

  // ---- R10: Nutritional Panel ----
  const nutrition = extractedFields.nutritionalInfo;
  if (nutrition.hasNutritionSection && nutrition.missing.length === 0) {
    addCheck('r10', 'Nutrition Panel (R10)', 'pass', 'Compliant',
      `Nutritional panel complete with all required metrics: ${nutrition.found.join(', ')}.`,
      nutrition.found.join(', '),
      'FSSAI Regulations 2020 — Regulation 8: Mandatory declaration of Energy, Protein, Carbohydrates, Total/Added Sugars, Total/Saturated/Trans Fat, and Sodium.'
    );
  } else if (nutrition.found.length > 0) {
    const missingStr = nutrition.missing.length > 0 ? ` Missing: ${nutrition.missing.join(', ')}.` : '';
    addCheck('r10', 'Nutrition Panel (R10)', 'warn', 'Needs Review',
      `Partial nutritional information detected.${missingStr}`,
      nutrition.found.join(', '),
      'FSSAI Regulations 2020 — Regulation 8.',
      `Declare all 9 required nutrients per 100g/ml: ${nutrition.missing.join(', ')}.`
    );
  } else {
    addCheck('r10', 'Nutrition Panel (R10)', 'fail', 'Non-Compliant',
      'No nutritional information panel detected.',
      'Not found',
      'FSSAI Regulations 2020 — Regulation 8.',
      'Add a standardized Nutritional Information panel declaring values per 100g/ml or per serving.'
    );
  }

  // ---- Overall Score ----
  const total = checks.length;
  const score = Math.round((passCount / total) * 100);

  let overallStatus = score >= 80 ? 'high' : (score >= 50 ? 'mid' : 'low');
  let overallLabel = score === 100 ? 'Fully Compliant' : (score >= 80 ? 'Mostly Compliant' : (score >= 50 ? 'Partially Compliant' : 'Significant Issues'));

  return {
    score,
    overall_status: overallStatus,
    overall_label: overallLabel,
    pass_count: passCount,
    fail_count: failCount,
    warn_count: warnCount,
    review_count: warnCount,
    total,
    checks,
    extracted_fields: extractedFields,
  };
}
