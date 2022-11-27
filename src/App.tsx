import React, { useRef, useEffect } from 'react';
import './App.css';
import Launchpad from './components/Launchpad/Launchpad';
const bg = require("../src/components/Launchpad/bg.jpeg");

function App() {

  document.body.style.background = `url(${bg})`

  return (
    <div className="App">
      <div className="container">
        <Launchpad />
      </div>

    </div>
  );
}

export default App;
