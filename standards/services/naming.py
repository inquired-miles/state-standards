"""Proxy naming service using optional OpenAI."""
import json
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI  # modern client
    _HAS_OPENAI_CLIENT = True
except Exception:  # pragma: no cover
    OpenAI = None
    _HAS_OPENAI_CLIENT = False
try:
    import openai  # legacy fallback
except Exception:  # pragma: no cover
    openai = None

from .base import BaseService
from ..models import ProxyStandard


class ProxyNamingService(BaseService):
    def __init__(self):
        super().__init__()
        self.client = None
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                if _HAS_OPENAI_CLIENT and OpenAI is not None:
                    self.client = OpenAI(api_key=api_key)
                elif openai is not None:
                    openai.api_key = api_key
                    self.client = openai
                logger.info("ProxyNamingService OpenAI client initialized: %s", bool(self.client))
        except Exception:
            self.client = None

    def name_proxy(self, proxy: ProxyStandard) -> Dict[str, str]:
        """Generate title and description for a proxy standard based on its source type"""
        try:
            # Get member texts based on source type
            member_texts = self._get_member_texts(proxy)
            
            if not member_texts:
                logger.warning(f"No member texts found for proxy {proxy.proxy_id}")
                return self._get_fallback_naming(proxy)
            
            # Generate appropriate prompt based on source type
            prompt = self._generate_prompt(proxy, member_texts)
            print(f"Generated prompt for {proxy.proxy_id}:")
            print(prompt)
            
            if not self.client:
                logger.warning("ProxyNamingService: OpenAI client unavailable, using fallback for %s", proxy.proxy_id)
                return self._get_fallback_naming(proxy)
            
            return self._call_openai_api(proxy, prompt)
            
        except Exception as e:
            logger.error(f"Error in name_proxy for {proxy.proxy_id}: {str(e)}")
            return self._get_fallback_naming(proxy)
    
    def _get_member_texts(self, proxy: ProxyStandard) -> list:
        """Extract texts from proxy members based on source type"""
        if proxy.source_type == 'atoms':
            return self._get_atom_texts(proxy)
        elif proxy.source_type == 'standards':
            return self._get_standard_texts(proxy)
        else:
            logger.warning(f"Unknown source type '{proxy.source_type}' for proxy {proxy.proxy_id}")
            return []
    
    def _get_atom_texts(self, proxy: ProxyStandard) -> list:
        """Extract texts from StandardAtoms"""
        atoms_all = list(proxy.member_atoms.all())
        
        # Put medoid first if it exists
        if proxy.medoid_atom_id:
            atoms_all.sort(key=lambda a: 0 if a.id == proxy.medoid_atom_id else 1)
        
        # Remove duplicates
        seen = set()
        unique_atoms = []
        for atom in atoms_all:
            key = " ".join((atom.text or "").split()).lower()
            if key in seen or not atom.text:
                continue
            seen.add(key)
            unique_atoms.append(atom.text)
        
        return unique_atoms
    
    def _get_standard_texts(self, proxy: ProxyStandard) -> list:
        """Extract texts from Standards"""
        standards_all = list(proxy.member_standards.all())
        
        # Put medoid first if it exists
        if proxy.medoid_standard_id:
            standards_all.sort(key=lambda s: 0 if s.id == proxy.medoid_standard_id else 1)
        
        # Remove duplicates and combine title + description
        seen = set()
        unique_standards = []
        for standard in standards_all:
            # Combine title and description for richer content
            text_parts = []
            if standard.title:
                text_parts.append(standard.title)
            if standard.description:
                text_parts.append(standard.description)
            
            full_text = " - ".join(text_parts)
            if not full_text:
                continue
                
            key = " ".join(full_text.split()).lower()
            if key in seen:
                continue
            seen.add(key)
            unique_standards.append(full_text)
        
        return unique_standards
    
    def _generate_prompt(self, proxy: ProxyStandard, member_texts: list) -> str:
        """Generate appropriate prompt based on source type"""
        # Build the content list
        content_text = "\n".join(f"{i+1}. {text}" for i, text in enumerate(member_texts))
        
        if proxy.source_type == 'standards':
            prompt = (
                "You will be given a group of curriculum 'Standards'; these are complete educational standards that students must master. "
                "These standards are similar to each other, but we need your expert advice on how to best name them and describe them all together as a single proxy standard. "
                "Your job is to: 1) A precise title (3-7 words) 2) Our proxy standard description / definition, this should be written like a standard. "
                "As you write this standard focus on the common learning objective, and the key concepts that students must master.\n\n"
                f"Standards:\n{content_text}"
            )
        else:  # atoms or fallback
            prompt = (
                "You will be given a group of curriculum 'Atoms'; these are elements of a curriculum unit that students must master. "
                "These atoms are similar to each other, but we need your expert advice on how to best name them and describe them all together as a single proxy standard. "
                "Your job is to: 1) A precise title (3-7 words) 2) Our proxy standard description / definition, this should be written like a standard. "
                "As you write this standard focus on the common learning objective, and the key concepts that students must master.\n\n"
                f"Atoms:\n{content_text}"
            )
        
        return prompt
    
    def _get_fallback_naming(self, proxy: ProxyStandard) -> Dict[str, str]:
        """Generate fallback naming based on source type"""
        medoid_text = self._get_medoid_text(proxy)
        
        return {
            "title": f"Standard {proxy.proxy_id}",
            "description": medoid_text[:100] if medoid_text else f"Proxy standard {proxy.proxy_id}"
        }
    
    def _get_medoid_text(self, proxy: ProxyStandard) -> str:
        """Get medoid text based on source type"""
        if proxy.source_type == 'atoms' and proxy.medoid_atom:
            return proxy.medoid_atom.text or ""
        elif proxy.source_type == 'standards' and proxy.medoid_standard:
            parts = []
            if proxy.medoid_standard.title:
                parts.append(proxy.medoid_standard.title)
            if proxy.medoid_standard.description:
                parts.append(proxy.medoid_standard.description)
            return " - ".join(parts)
        return ""
    
    def _call_openai_api(self, proxy: ProxyStandard, prompt: str) -> Dict[str, str]:
        """Call OpenAI API with the generated prompt"""
        try:
            # Prefer Responses API (K/SON JSON schema) if available
            if hasattr(self.client, 'responses'):
                schema = {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "A precise proxy standard title",
                            "minLength": 3,
                        },
                        "description": {
                            "type": "string",
                            "description": "A one-sentence proxy standard description, written like a standard.",
                            "minLength": 8,
                        }
                    },
                    "required": ["title", "description"],
                    "additionalProperties": False
                }
                resp = self.client.responses.create(
                    model="gpt-4.1",
                    input=[
                        {"role": "system", "content": [{"type": "input_text", "text": "You are an education curriculum expert. Provide recommendations and guidance on curriculum design and improvement as an education curriculum expert."}]},
                        {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "curriculum_summary",
                            "strict": True,
                            "schema": schema
                        }
                    },
                    temperature=0
                )
                content = getattr(resp, 'output_text', None)
                if not content:
                    try:
                        # Best-effort for other client shapes
                        content = resp.output[0].content[0].text  # type: ignore[attr-defined]
                    except Exception:
                        content = None
                if not content:
                    raise ValueError("Empty response from Responses API")

            # Fallback: Chat Completions with JSON enforced
            elif hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You are an education curriculum expert."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0
                )
                content = resp.choices[0].message.content

            # Legacy fallback
            else:
                resp = self.client.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an education curriculum expert."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0
                )
                content = resp["choices"][0]["message"]["content"]

            data = json.loads(content)
            logger.info("ProxyNamingService: named %s => %s", proxy.proxy_id, data)
            return {"title": data.get("title") or "", "description": data.get("description") or ""}
        except Exception as e:
            logger.warning("ProxyNamingService error for %s: %s", proxy.proxy_id, str(e))
            return self._get_fallback_naming(proxy)



