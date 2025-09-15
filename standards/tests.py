from django.test import TestCase
from .models import State, Standard, StandardAtom


class NormalizationTests(TestCase):
    def test_parser_formats(self):
        from .utils.normalization import StandardParser
        p = StandardParser()
        ca = p.parse('SS.3.4.1', 't', 3)
        tx = p.parse('3.4(A)', 't', 3)
        gen = p.parse('RANDOM-123', 't', 3)
        self.assertEqual(ca.format_type, 'california')
        self.assertEqual(tx.format_type, 'texas')
        self.assertEqual(gen.format_type, 'generic')


class AtomizerTests(TestCase):
    def setUp(self):
        self.state = State.objects.create(code='CA', name='California')
        # minimal subject/grade to satisfy FK; we bypass M2M for brevity
        self.standard = Standard.objects.create(
            state=self.state,
            subject_area_id=1,
            code='SS.3.4.1',
            title='Community Helpers',
            description='Students identify and describe community helpers including police and firefighters.',
        )

    def test_atomizer_splits(self):
        from .services.atomizer import AtomizationService
        svc = AtomizationService()
        atoms = svc.generate_atoms_for_standard(self.standard, use_gpt=False)
        self.assertTrue(len(atoms) >= 1)
        self.assertTrue(any('identify' in a.text.lower() or 'describe' in a.text.lower() for a in atoms))


class CoverageRollupTests(TestCase):
    def setUp(self):
        self.state = State.objects.create(code='CA', name='California')
        self.standard = Standard.objects.create(
            state=self.state,
            subject_area_id=1,
            code='SS.3.1',
            title='Test',
            description='Test standard'
        )
        self.atoms = [
            StandardAtom.objects.create(standard=self.standard, atom_code='SS.3.1-A', text='Atom 1', embedding=[1.0, 0.0]),
            StandardAtom.objects.create(standard=self.standard, atom_code='SS.3.1-B', text='Atom 2', embedding=[1.0, 0.0]),
            StandardAtom.objects.create(standard=self.standard, atom_code='SS.3.1-C', text='Atom 3', embedding=[0.0, 1.0]),
        ]

    def test_partial_minimal_status(self):
        from .models import CurriculumDocument, CoverageDetail, StandardCoverage
        from .services.coverage_v2 import CoverageAnalyzer
        cur = CurriculumDocument.objects.create(name='Cur', content='x')
        # Mock coverage: 2 of 3 above threshold
        CoverageDetail.objects.create(curriculum=cur, atom=self.atoms[0], similarity_score=0.9, is_covered=True)
        CoverageDetail.objects.create(curriculum=cur, atom=self.atoms[1], similarity_score=0.8, is_covered=True)
        CoverageDetail.objects.create(curriculum=cur, atom=self.atoms[2], similarity_score=0.4, is_covered=False)
        CoverageAnalyzer().calculate_standard(cur)
        sc = StandardCoverage.objects.get(curriculum=cur, standard=self.standard)
        self.assertEqual(sc.total_atoms, 3)
        self.assertEqual(sc.covered_atoms, 2)
        self.assertTrue(66.0 <= sc.coverage_percentage <= 67.0)
        self.assertEqual(sc.status, 'MINIMAL')
