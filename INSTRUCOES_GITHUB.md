# Instruções para Enviar ao GitHub

## Arquivos Modificados/Adicionados

### 📁 Arquivos na pasta `bridgekit/`
- `config.py` - Atualizado com suporte multi-provider
- `providers.py` - **NOVO** - Factory para clientes de diferentes providers
- `reviewer.py` - Atualizado com parâmetros provider/model
- `planner.py` - Atualizado com parâmetros provider/model  
- `redteam.py` - Atualizado com parâmetros provider/model
- `search.py` - Atualizado com parâmetros provider/model

### 📁 Arquivos na raiz
- `pyproject.toml` - Atualizado com novas dependências
- `README.md` - Atualizado com documentação multi-provider

## 🚀 Como Enviar

1. **Faça backup dos arquivos originais** (se necessário)
2. **Substitua os arquivos** no repositório local:
   ```bash
   # Copiar arquivos da pasta bridgekit/
   cp Enviar/bridgekit/* seu-repositorio/bridgekit/
   
   # Copiar arquivos da raiz
   cp Enviar/pyproject.toml seu-repositorio/
   cp Enviar/README.md seu-repositorio/
   ```

3. **Instale as novas dependências**:
   ```bash
   pip install openai>=1.0.0 google-generativeai>=0.3.0
   ```

4. **Teste a instalação**:
   ```bash
   python -c "from bridgekit import evaluate, plan, ask, redteam; print('✅ Import successful')"
   ```

5. **Commit e push**:
   ```bash
   git add .
   git commit -m "feat: Add multi-provider support (Anthropic, OpenAI, Gemini)"
   git push origin main
   ```

## ✨ Novas Funcionalidades

- **Suporte a 3 providers**: Anthropic, OpenAI, Google Gemini
- **Auto-detecção**: Provider detectado automaticamente pelo nome do modelo
- **Backward compatibility**: Código existente continua funcionando
- **API Keys**: Suporte para `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`

## 📖 Exemplos de Uso

```python
from bridgekit import evaluate, plan, ask, redteam

# Diferentes providers
print(evaluate("texto", provider="openai"))
print(plan("pergunta", provider="gemini"))

# Modelos específicos
print(redteam("texto", model="gpt-4-turbo"))
print(ask("pergunta", source="docs/", model="claude-3-opus"))

# Padrão (backward compatible)
print(evaluate("texto"))  # Usa Anthropic
```

## 🔧 Variáveis de Ambiente

```bash
export ANTHROPIC_API_KEY=sua_chave_aqui
export OPENAI_API_KEY=sua_chave_aqui  
export GOOGLE_API_KEY=sua_chave_aqui
```

---

**Pronto!** O BridgeKit agora suporta múltiplos providers de IA. 🎉
