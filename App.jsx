import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import Header from './Components/Header'
import ImageDisplay from './Components/Imagedisplay'
import Buttons from './Components/Buttons'

function App() {
  const [imageUrl, setImageUrl] = useState(null)

  const handleGenerate= async () => {
    console.log("Button clicked");

   
    
      try {
        await fetch('http://127.0.0.1:5000/generate-image');
        const imageUrl = 'http://127.0.0.1:5000/static/generated/generated.png?timestamp=' + new Date().getTime();
        setImageUrl(imageUrl);
      } catch (err) {
        console.error('Failed to fetch image:', err);
      }
    };
    

   
    
    

  

  const handleDownload = () => {
    if (!imageUrl) return;
  
    const filename = 'generated.png';
    const downloadUrl = `http://127.0.0.1:5000/api/download/${filename}`;
    
  
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  

  return(
    <>
      <Header/>

      <ImageDisplay imageUrl={imageUrl}/>

      <Buttons OnGenerate={handleGenerate} OnDownload={handleDownload}/>

      
    </>

  )
  
    
  
}

export default App
