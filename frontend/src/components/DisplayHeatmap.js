import React from 'react';

const DisplayHeatmap = ({base64String}) => {
    const imageSource = `data:image/png;base64,${base64String}`;

    return (
        <div className="image-display-container">
            <img src={imageSource} alt="Base64 Image"/>
        </div>
    );
};

export default DisplayHeatmap;
