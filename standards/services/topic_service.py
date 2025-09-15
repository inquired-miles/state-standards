"""Topic extraction (GPT optional) and verification via embeddings."""
import json
import os
from typing import Dict, List

import numpy as np

try:
    import openai
except Exception:  # pragma: no cover
    openai = None

from .base import BaseService
from .embedding import EmbeddingService
from ..models import State, StandardAtom


class TopicAnalyzer(BaseService):
    def __init__(self):
        super().__init__()
        self.client = None
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and openai:
                openai.api_key = api_key
                self.client = openai
        except Exception:
            self.client = None
        self.embedding_service = EmbeddingService()

    def extract_topics(self, sample_per_state: int = 10, max_states: int = 25) -> Dict:
        samples: List[str] = []
        for state in State.objects.all()[:max_states]:
            texts = list(StandardAtom.objects.filter(standard__state=state).values_list('text', flat=True)[:sample_per_state])
            samples.extend([f"{state.code}: {t}" for t in texts])
        if not self.client:
            return {"topics": []}
        prompt = (
            "Analyze these Grade 3 Social Studies atom texts across states. Identify 15-20 major topics, "
            "estimate coverage %, provide keywords and an example. Return JSON {\"topics\": [...]}\n\n" + "\n".join(samples[:100])
        )
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are an education curriculum expert."}, {"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception:
            return {"topics": []}

    def verify_topic(self, name: str, keywords: List[str] = None) -> Dict:
        query = name + (" " + " ".join(keywords) if keywords else "")
        qvec = np.array(self.embedding_service.generate_embedding(query) or [])
        state_details = {}
        total_matches = 0
        for state in State.objects.all():
            matches = []
            atoms = StandardAtom.objects.filter(standard__state=state, embedding__isnull=False)
            for a in atoms:
                avec = np.array(a.embedding)
                sim = float((qvec @ avec) / ((np.linalg.norm(qvec) * np.linalg.norm(avec)) + 1e-9)) if qvec.size else 0.0
                if sim >= 0.75:
                    matches.append({'standard': a.atom_code, 'text': a.text[:100], 'similarity': sim})
                    total_matches += 1
            state_details[state.code] = {
                'has_topic': len(matches) > 0,
                'match_count': len(matches),
                'top_matches': sorted(matches, key=lambda x: x['similarity'], reverse=True)[:3],
            }
        states_with_topic = sum(1 for v in state_details.values() if v['has_topic'])
        coverage_pct = (states_with_topic / max(1, len(state_details))) * 100
        return {
            'topic': name,
            'coverage_percentage': coverage_pct,
            'states_covered': states_with_topic,
            'total_matches': total_matches,
            'state_details': state_details,
        }

    def analyze_all(self) -> Dict:
        extracted = self.extract_topics()
        verified = []
        for t in extracted.get('topics', []):
            v = self.verify_topic(t.get('name', ''), t.get('keywords', []))
            t['verified_coverage'] = v['coverage_percentage']
            t['verified_states'] = v['states_covered']
            verified.append(t)
        verified.sort(key=lambda x: x.get('verified_coverage', 0), reverse=True)
        universal = [x for x in verified if x.get('verified_coverage', 0) >= 80]
        common = [x for x in verified if 40 <= x.get('verified_coverage', 0) < 80]
        regional = [x for x in verified if x.get('verified_coverage', 0) < 40]
        return {
            'all_topics': verified,
            'universal': universal,
            'common': common,
            'regional': regional,
            'summary': {
                'total_topics': len(verified),
                'universal_count': len(universal),
                'common_count': len(common),
                'regional_count': len(regional),
            }
        }



