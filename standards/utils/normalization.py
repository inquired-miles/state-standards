import re
from dataclasses import dataclass
from typing import List


@dataclass
class ParsedStandard:
    original_code: str
    grade: int
    parts: List[str]
    text: str
    format_type: str


class FormatDetector:
    PATTERNS = {
        'california': r'^([A-Z]+)\.(\d+)\.(\d+)\.?((?:\d+)|(?:[A-Za-z]))?$',
        'texas': r'^(\d+)\.(\d+)\(([A-Z])\)',
        'common_core': r'^CCSS\.(\d+)\.([A-Z]+)\.([A-Z])\.(\d+)',
        'florida': r'^([A-Z]+)\.(\d+)\.([A-Z])\.(\d+)\.?((?:\d+)|(?:[A-Za-z]))?$',
        'complex': r'^(\d+)\.([a-z])\.([a-z])\.(\d+)\.([a-z])',
    }

    def detect_format(self, sample_codes: List[str]) -> str:
        scores = {k: 0 for k in self.PATTERNS.keys()}
        for code in sample_codes:
            for name, pat in self.PATTERNS.items():
                if re.match(pat, code or ''):
                    scores[name] += 1
        detected = max(scores.items(), key=lambda x: x[1])
        return detected[0] if detected[1] > 0 else 'generic'


class StandardParser:
    def parse(self, raw_code: str, text: str, default_grade: int = 3) -> ParsedStandard:
        # California: SS.3.4.1
        m = re.match(r'^([A-Z]+)\.(\d+)\.(\d+)\.?((?:\d+)|(?:[A-Za-z]))?$', raw_code or '')
        if m:
            parts = [m.group(3)]
            if m.group(4):
                parts.append(m.group(4))
            return ParsedStandard(raw_code, int(m.group(2)), parts, text, 'california')

        # Texas: 3.4(A)
        m = re.match(r'^(\d+)\.(\d+)\(([A-Z])\)', raw_code or '')
        if m:
            return ParsedStandard(raw_code, int(m.group(1)), [m.group(2), m.group(3)], text, 'texas')

        # Common-core like
        m = re.match(r'^CCSS\.(\d+)\.([A-Z]+)\.([A-Z])\.(\d+)', raw_code or '')
        if m:
            return ParsedStandard(raw_code, int(m.group(1)), [m.group(2), m.group(3), m.group(4)], text, 'common_core')

        # Complex: 4.a.a.3.a
        m = re.match(r'^(\d+)\.(.*)$', raw_code or '')
        if m:
            rest = raw_code.split('.')[1:] if raw_code else []
            return ParsedStandard(raw_code, int(m.group(1)), rest, text, 'complex')

        # generic
        return ParsedStandard(raw_code, default_grade, [raw_code], text, 'generic')



