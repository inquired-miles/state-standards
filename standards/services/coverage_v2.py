"""Coverage analysis pipeline: curriculum -> atoms -> standards -> states."""
from typing import Dict, List

import numpy as np
from django.db import transaction
from django.db.models import Avg
from pgvector.django import CosineDistance

from .base import BaseService
from .embedding import EmbeddingService
from ..models import (
    CurriculumDocument, CoverageDetail, StandardCoverage, StateCoverage,
    StandardAtom, Standard, State
)


class CoverageAnalyzer(BaseService):
    THRESHOLDS = dict(full=0.90, partial=0.70, minimal=0.40)
    SIMILARITY = 0.75

    def __init__(self):
        super().__init__()
        self.embedder = EmbeddingService()

    @transaction.atomic
    def analyze_curriculum(self, curriculum: CurriculumDocument) -> Dict:
        self.calculate_atomic(curriculum)
        self.calculate_standard(curriculum)
        self.calculate_state(curriculum)
        curriculum.processed = True
        curriculum.save(update_fields=['processed'])
        return self.report(curriculum)

    def calculate_atomic(self, curriculum: CurriculumDocument):
        # Ensure curriculum embedding
        if not curriculum.embedding:
            emb = self.embedder.generate_embedding(curriculum.content)
            if not emb:
                raise ValueError("Failed to generate curriculum embedding")
            curriculum.embedding = emb
            curriculum.save(update_fields=['embedding'])

        # Compute similarity to all atoms via pgvector
        atoms = StandardAtom.objects.filter(embedding__isnull=False).select_related('standard')
        # Iterate in chunks to avoid memory pressure
        to_create: List[CoverageDetail] = []
        for atom in atoms.iterator(chunk_size=1000):
            # Using Python cosine for portability if needed, else database-side with annotation
            # Here, fetch similarity via db function to avoid numpy where not available
            # Fallback to Python compute if annotation import fails
            try:
                # Simplest path: compute in Python
                sim = float(np.dot(np.array(curriculum.embedding), np.array(atom.embedding)) /
                            ( np.linalg.norm(curriculum.embedding) * np.linalg.norm(atom.embedding) ))
            except Exception:
                sim = 0.0

            to_create.append(CoverageDetail(
                curriculum=curriculum,
                atom=atom,
                similarity_score=sim,
                is_covered=sim >= self.SIMILARITY,
            ))
            if len(to_create) >= 1000:
                CoverageDetail.objects.bulk_create(to_create, ignore_conflicts=True)
                to_create = []
        if to_create:
            CoverageDetail.objects.bulk_create(to_create, ignore_conflicts=True)

    def calculate_standard(self, curriculum: CurriculumDocument):
        # Roll atoms -> standard
        # Compute per-standard totals and covered counts
        details = CoverageDetail.objects.filter(curriculum=curriculum).select_related('atom__standard')
        per_standard: Dict[str, Dict[str, int]] = {}
        for d in details.iterator(chunk_size=2000):
            sid = str(d.atom.standard_id)
            entry = per_standard.setdefault(sid, dict(total=0, covered=0))
            entry['total'] += 1
            if d.is_covered:
                entry['covered'] += 1

        records: List[StandardCoverage] = []
        id_to_standard: Dict[str, Standard] = {str(s.id): s for s in Standard.objects.filter(id__in=per_standard.keys())}
        for sid, data in per_standard.items():
            total = data['total']
            covered = data['covered']
            pct = (covered / total) * 100 if total else 0.0
            if pct >= self.THRESHOLDS['full'] * 100:
                status = 'FULL'
            elif pct >= self.THRESHOLDS['partial'] * 100:
                status = 'PARTIAL'
            elif pct >= self.THRESHOLDS['minimal'] * 100:
                status = 'MINIMAL'
            else:
                status = 'NONE'
            records.append(StandardCoverage(
                curriculum=curriculum,
                standard=id_to_standard[sid],
                total_atoms=total,
                covered_atoms=covered,
                coverage_percentage=pct,
                status=status,
            ))
        StandardCoverage.objects.bulk_create(records, ignore_conflicts=True)

    def calculate_state(self, curriculum: CurriculumDocument):
        per_state: Dict[str, Dict[str, int]] = {}
        std_cov = StandardCoverage.objects.filter(curriculum=curriculum).select_related('standard__state')
        for sc in std_cov.iterator(chunk_size=2000):
            state = sc.standard.state
            s = per_state.setdefault(state.code, dict(total=0, full=0, partial=0, minimal=0, none=0, state=state))
            s['total'] += 1
            if sc.status == 'FULL':
                s['full'] += 1
            elif sc.status == 'PARTIAL':
                s['partial'] += 1
            elif sc.status == 'MINIMAL':
                s['minimal'] += 1
            else:
                s['none'] += 1

        records: List[StateCoverage] = []
        for code, data in per_state.items():
            total = data['total']
            overall = (
                data['full'] * 100 + data['partial'] * 80 + data['minimal'] * 50 + data['none'] * 0
            ) / total if total else 0.0
            is_marketable = (data['full'] / total) >= 0.80 if total else False
            records.append(StateCoverage(
                curriculum=curriculum,
                state=data['state'],
                total_standards=total,
                full_coverage_count=data['full'],
                partial_coverage_count=data['partial'],
                minimal_coverage_count=data['minimal'],
                none_coverage_count=data['none'],
                overall_percentage=overall,
                is_marketable=is_marketable,
            ))
        StateCoverage.objects.bulk_create(records, ignore_conflicts=True)

    def report(self, curriculum: CurriculumDocument) -> Dict:
        # Quick wins: states in 70-79% overall, list top partial standards and sample missing atoms
        state_qs = StateCoverage.objects.filter(curriculum=curriculum)
        quick_win_states = state_qs.filter(overall_percentage__gte=70, overall_percentage__lt=80)
        quick_wins = []
        for sc in quick_win_states[:5]:
            partials = StandardCoverage.objects.filter(curriculum=curriculum, standard__state=sc.state, status='PARTIAL')[:3]
            for p in partials:
                missing = CoverageDetail.objects.filter(curriculum=curriculum, atom__standard=p.standard, is_covered=False).select_related('atom')[:2]
                quick_wins.append({
                    'state': sc.state.code,
                    'standard_code': p.standard.code,
                    'current_coverage': p.coverage_percentage,
                    'missing_atoms': [m.atom.text for m in missing],
                })

        total_states = State.objects.count()
        analyzed_states = state_qs.count()
        avg_coverage = state_qs.aggregate(avg=Avg('overall_percentage'))['avg'] or 0
        return {
            'curriculum': curriculum.id,
            'summary': {
                'marketable_states': state_qs.filter(is_marketable=True).count(),
                'total_states': analyzed_states,
                'avg_coverage': avg_coverage,
            },
            'quick_wins': quick_wins,
            'states': list(state_qs.values('state__code', 'overall_percentage', 'full_coverage_count', 'partial_coverage_count', 'minimal_coverage_count', 'none_coverage_count', 'is_marketable')),
        }



