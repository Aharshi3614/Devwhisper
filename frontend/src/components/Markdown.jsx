import { useEffect } from 'react'
import Prism from 'prismjs'

// Import Prism language components for syntax highlighting support
import 'prismjs/components/prism-python'
import 'prismjs/components/prism-javascript'
import 'prismjs/components/prism-typescript'
import 'prismjs/components/prism-jsx'
import 'prismjs/components/prism-tsx'
import 'prismjs/components/prism-css'
import 'prismjs/components/prism-markup' // HTML/XML
import 'prismjs/components/prism-json'
import 'prismjs/components/prism-bash'
import 'prismjs/components/prism-markdown'
import 'prismjs/components/prism-sql'

// Heuristic language detector for code blocks missing a language specifier
const detectLanguage = (code, parsedLang) => {
  if (parsedLang) {
    const lang = parsedLang.toLowerCase()
    // Map common aliases
    if (lang === 'py') return 'python'
    if (lang === 'js') return 'javascript'
    if (lang === 'ts') return 'typescript'
    if (lang === 'html') return 'markup'
    if (lang === 'sh') return 'bash'
    return lang
  }

  // Fallback heuristic based on common code signatures
  if (code.includes('import ') && (code.includes('def ') || code.includes('print('))) {
    return 'python'
  }
  if (code.includes('def ') || code.includes('self.')) {
    return 'python'
  }
  if (code.includes('const ') || code.includes('let ') || code.includes('function ') || code.includes('console.log')) {
    return 'javascript'
  }
  if (code.includes('import React') || code.includes('export default') || code.includes('<div') || code.includes('className=')) {
    return 'jsx'
  }
  if (code.includes('<!DOCTYPE html>') || code.includes('<html') || code.includes('<body>')) {
    return 'markup'
  }
  if (code.includes('{') && code.includes('}') && (code.includes('margin:') || code.includes('padding:') || code.includes('color:'))) {
    return 'css'
  }
  if (code.includes('select ') && code.includes('from ') && code.includes('where ')) {
    return 'sql'
  }

  return 'javascript' // default fallback
}

function Markdown({ text }) {
  // Re-highlight when Markdown content is fully updated
  useEffect(() => {
    Prism.highlightAll()
  }, [text])

  if (!text) return null

  // Ensure unclosed code blocks from live stream are closed properly
  const occurrences = (text.match(/```/g) || []).length
  const isUnclosed = occurrences % 2 !== 0

  let parsedText = text
  if (isUnclosed) {
    parsedText += '\n```'
  }

  const parts = parsedText.split(/(```[\s\S]*?```)/g)
  return parts.map((part, index) => {
    if (part.startsWith('```')) {
      const match = part.match(/```(\w*)\n([\s\S]*?)```/)
      const rawLanguage = match ? match[1] : ''
      const code = match ? match[2] : part.slice(3, -3)
      const language = detectLanguage(code, rawLanguage)

      // Get Prism grammar definition or default to javascript
      const grammar = Prism.languages[language] || Prism.languages.javascript
      const highlightedHtml = Prism.highlight(code.trim(), grammar, language)

      return (
        <pre key={index} className="code-block">
          {language && <span className="code-lang">{language}</span>}
          <code 
            className={`language-${language}`} 
            dangerouslySetInnerHTML={{ __html: highlightedHtml }} 
          />
        </pre>
      )
    }

    const inlineParts = part.split(/(`[^`\n]+`)/g)
    return (
      <span key={index}>
        {inlineParts.map((subPart, subIndex) => {
          if (subPart.startsWith('`') && subPart.endsWith('`')) {
            return (
              <code key={subIndex} className="inline-code">
                {subPart.slice(1, -1)}
              </code>
            )
          }
          return subPart
        })}
      </span>
    )
  })
}

export default Markdown
