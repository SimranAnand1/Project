import React, {useState} from "react";
import {OverlayTrigger, Tooltip} from 'react-bootstrap';

const Header = () => {
    const [showTooltip, setShowTooltip] = useState(false);

    const renderTooltip = (props) => (
        <Tooltip id="text-on-hover-tooltip" {...props}>
            <p>Image Captioning is the process of <strong>generating textual description</strong> of an image. It uses
                both Natural Language Processing and Computer Vision to generate the captions. These models are trained
                on Flickr30k dataset</p>
        </Tooltip>
    );

    return (
        <OverlayTrigger
            show={showTooltip}
            placement="bottom"
            delay={{show: 50, hide: 400}}
            overlay={renderTooltip}
        >
            <h1 className="text-center mt-3 mb-3"
                onMouseEnter={() => setShowTooltip(true)}
                onMouseLeave={() => setShowTooltip(false)}
            >
                Image Captioner
            </h1>
        </OverlayTrigger>
    );
};

export default Header;