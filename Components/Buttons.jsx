import React from 'react'
import '../Components/btncss.css'

function Buttons({OnGenerate, OnDownload}) {
  
    return (
      <div className='btnsclass'>
      <button type='button' onClick={OnGenerate}>Generate Image</button>
      <button type='button' onClick={OnDownload}>Download image</button>
      
          
      </div>
    )
  }
  
  export default Buttons