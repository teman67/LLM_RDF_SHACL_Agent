from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import requests

class BaseAgent(ABC):
    """Base class for all LLM agents"""
    
    def __init__(self, name: str, model_config: Dict[str, Any]):
        self.name = name
        self.model_config = model_config
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        provider = self.model_config['provider']
        
        if provider == "OpenAI":
            return OpenAI(api_key=self.model_config['api_key'])
        elif provider == "Anthropic (Claude)":
            return Anthropic(api_key=self.model_config['api_key'])
        elif provider == "Ollama (Self-hosted)":
            return None  # Ollama uses direct API calls
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Generic LLM calling method"""
        try:
            provider = self.model_config['provider']
            
            if provider == "OpenAI":
                response = self.client.chat.completions.create(
                    model=self.model_config['model'],
                    messages=[
                        {"role": "system", "content": system_prompt or self.get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.model_config['temperature'],
                    max_tokens=self.model_config.get('max_tokens', 4000)
                )
                return response.choices[0].message.content
                
            elif provider == "Anthropic (Claude)":
                response = self.client.messages.create(
                    model=self.model_config['model'],
                    max_tokens=self.model_config.get('max_tokens', 4000),
                    temperature=self.model_config['temperature'],
                    system=system_prompt or self.get_system_prompt(),
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            else:  # Ollama
                return self._call_ollama_api(prompt, system_prompt)
                
        except Exception as e:
            st.error(f"Error calling {self.name}: {str(e)}")
            return f"Error: {str(e)}"
    
    def _call_ollama_api(self, prompt: str, system_prompt: str = None) -> str:
        """Call Ollama API"""
        endpoint = self.model_config['endpoint']
        headers = {"Content-Type": "application/json"}
        
        if system_prompt:
            data = {
                "model": self.model_config['model'],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "temperature": self.model_config['temperature']
            }
        else:
            data = {
                "model": self.model_config['model'],
                "prompt": prompt,
                "stream": False,
                "temperature": self.model_config['temperature']
            }
        
        try:
            chat_url = f"{endpoint.rstrip('/')}/api/chat"
            response = requests.post(chat_url, json=data, headers=headers)
            
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                # Fallback to generate endpoint
                generate_url = f"{endpoint.rstrip('/')}/api/generate"
                generate_data = {"model": self.model_config['model'], "prompt": prompt, "temperature": self.model_config['temperature']}
                response = requests.post(generate_url, json=generate_data, headers=headers)
                
                if response.status_code == 200:
                    return response.json()["response"]
                else:
                    return f"Error: Ollama API returned {response.status_code}"
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent"""
        pass
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and return result"""
        pass