import React, { useState } from 'react';
import './App.css';

function App() {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [loading, setLoading] = useState(false);

  const handleParaphrase = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/paraphrase', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });
      
      const data = await response.json();
      setOutputText(data.paraphrased);
    } catch (error) {
      console.error('Error:', error);
      setOutputText('Error connecting to server');
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <div className="container">
        <h1>AI Paraphraser</h1>
        <div className="input-section">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter text to paraphrase..."
            rows={6}
          />
        </div>
        
        <button 
          onClick={handleParaphrase} 
          disabled={loading || !inputText}
          className="paraphrase-btn"
        >
          {loading ? 'Processing...' : 'Paraphrase'}
        </button>
        
        <div className="output-section">
          <textarea
            value={outputText}
            readOnly
            placeholder="Paraphrased text will appear here..."
            rows={6}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
