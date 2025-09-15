"""
Atomization services: split standards into atomic learning objectives and persist as StandardAtom.
"""
import re
import json
import os
from typing import List, Dict
from django.utils import timezone

try:
    import openai
except Exception:  # pragma: no cover - optional
    openai = None

from .base import BaseService
from ..models import Standard, StandardAtom


class ContentAtomizer(BaseService):
    """Rule-based with optional GPT-assisted atomization."""

    VERBS = [
        'identify', 'describe', 'explain', 'compare', 'contrast',
        'analyze', 'evaluate', 'create', 'understand', 'demonstrate',
        'apply', 'summarize', 'interpret', 'classify', 'predict',
        'investigate', 'examine', 'discuss', 'illustrate', 'organize'
    ]

    def __init__(self):
        super().__init__()
        # Initialize OpenAI if available
        self.openai_client = None
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and openai:
                openai.api_key = api_key
                self.openai_client = openai
        except Exception:
            self.openai_client = None

    def is_compound(self, text: str) -> bool:
        markers = [r'\band\b', r'including', r'such as', r';', r',\s+and\s+']
        has_markers = any(re.search(p, text.lower()) for p in markers)
        verbs = [v for v in self.VERBS if v in (text or '').lower()]
        return has_markers or len(verbs) > 1

    def simple_atomize(self, text: str) -> List[Dict[str, str]]:
        parts = re.split(r'\s+and\s+|\s+including\s+|;\s*', text or '')
        parts = [p.strip() for p in parts if len(p.strip().split()) > 3]
        if not parts:
            return [{"code": "A", "text": text}]
        return [{"code": chr(65 + i), "text": p} for i, p in enumerate(parts)]

    def atomize_with_gpt(self, text: str) -> List[Dict[str, str]]:
        if not self.openai_client:
            return self.simple_atomize(text)

        prompt = (
            "Break this education standard into atomic learning objectives. Each atom is a single, "
            "testable objective. Rules: (1) One objective per atom (2) Preserve grade-appropriate language "
            "(3) Keep related context together. Return JSON array like: "
            "[{\"code\":\"A\",\"text\":\"...\"},{\"code\":\"B\",\"text\":\"...\"}]\n\n"
            f"Standard: \"{text}\""
        )
        try:
            resp = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # small-fast default; adjust if needed
                messages=[
                    {"role": "system", "content": "You are an education standards expert."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and 'atoms' in data:
                return data['atoms']
        except Exception:
            pass
        return self.simple_atomize(text)


class AtomizationService(BaseService):
    def __init__(self):
        super().__init__()
        self.atomizer = ContentAtomizer()

    def _next_atom_code(self, base_code: str, index: int) -> str:
        return f"{base_code}-{chr(65 + index)}"

    def _base_code(self, standard: Standard) -> str:
        state_prefix = ""
        try:
            if getattr(standard, 'state', None) and getattr(standard.state, 'code', None):
                state_prefix = f"{standard.state.code}."
        except Exception:
            state_prefix = ""
        return f"{state_prefix}{standard.code or str(standard.id)}"

    def generate_atoms_for_standard(self, standard: Standard, use_gpt: bool = False) -> List[StandardAtom]:
        text = standard.description or standard.title or ''
        base_code = self._base_code(standard)

        # Helper to allocate a unique atom_code without reassigning existing atoms
        def allocate_unique_code(preferred: str) -> str:
            code = preferred
            suffix_idx = 1
            while StandardAtom.objects.filter(atom_code=code).exclude(standard=standard).exists():
                code = f"{preferred}-{suffix_idx}"
                suffix_idx += 1
            return code

        # If already has atoms, skip idempotently
        if standard.atoms.exists():
            return list(standard.atoms.all())

        # If not compound, create single atom mirroring the standard
        if not self.atomizer.is_compound(text):
            unique_code = allocate_unique_code(base_code)
            atom, created = StandardAtom.objects.get_or_create(
                atom_code=unique_code,
                defaults={
                    'standard': standard,
                    'text': text,
                    'method': 'content' if text else 'structural',
                    'embedding_generated_at': None,
                }
            )
            if not created and atom.standard_id != standard.id:
                # Shouldn't happen due to allocate_unique_code, but guard anyway
                unique_code = allocate_unique_code(f"{base_code}-DUP")
                atom = StandardAtom.objects.create(
                    standard=standard,
                    atom_code=unique_code,
                    text=text,
                    method='content' if text else 'structural',
                    embedding_generated_at=None,
                )
            # Mark standard as atomic
            if not standard.is_atomic:
                standard.is_atomic = True
                standard.save(update_fields=['is_atomic'])
            return [atom]

        # Compound: split
        atoms_spec = self.atomizer.atomize_with_gpt(text) if use_gpt else self.atomizer.simple_atomize(text)
        created: List[StandardAtom] = []
        for i, spec in enumerate(atoms_spec):
            preferred_code = self._next_atom_code(base_code, i)
            code = allocate_unique_code(preferred_code)
            atom, created_flag = StandardAtom.objects.get_or_create(
                atom_code=code,
                defaults={
                    'standard': standard,
                    'text': spec.get('text', '').strip() or text,
                    'method': 'gpt' if use_gpt else 'content',
                }
            )
            if not created_flag and atom.standard_id != standard.id:
                # Create a new unique code for this standard
                new_code = allocate_unique_code(f"{preferred_code}-DUP")
                atom = StandardAtom.objects.create(
                    standard=standard,
                    atom_code=new_code,
                    text=spec.get('text', '').strip() or text,
                    method='gpt' if use_gpt else 'content',
                )
            created.append(atom)

        return created



