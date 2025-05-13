import React from 'react'
import '../Components/imagediscss.css'


function ImageDisplay({imageUrl}) {
    return (
      <div className='imageclass'>
      {imageUrl ? (
        <img src={imageUrl} alt='generatedimage' style={{width:'500px', height:'500px'}}/>
      ) : (
        <p>no image generated yet</p>
      )}
      
      
          
      </div>
    )
  }
  
  export default ImageDisplay