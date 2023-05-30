import React, {useState, useRef} from "react";
import {toast} from "react-toastify";
import {convertBytesToMb} from "../utils/helpers";


const InputImage = ({handleSelectImage}) => {
    const [dragActive, setDragActive] = useState(false);
    const [previewUrl, setPreviewUrl] = useState(null);
    const inputRef = useRef(null);

    const handleFile = (file) => {
        if (!file.type.startsWith("image")) {
            toast.error("Selected file is not an image! Please select an image file.");
            return;
        }

        if (file.size > Number(process.env.REACT_APP_MAX_IMAGE_SIZE)) {
            toast.error(`Image size should be less than ${convertBytesToMb(process.env.REACT_APP_MAX_IMAGE_SIZE)} MB`);
            return;
        }

        const reader = new FileReader();
        reader.onload = e => {
            setPreviewUrl(e.target.result);
        };
        reader.readAsDataURL(file);

        handleSelectImage(file);
    };

    // handle drag events
    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    // triggers when file is dropped
    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    };

    // triggers when file is selected with click
    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    };

    return (
        <form id="form-file-upload" onDragEnter={handleDrag} onSubmit={(e) => e.preventDefault()}>
            <input id="input-file-upload" ref={inputRef} type="file" accept="image/*" onChange={handleChange}/>
            <label id="label-file-upload" htmlFor="input-file-upload" className={dragActive ? "drag-active" : ""}>
                <div className="file-box">
                    {previewUrl ? (
                        <img className="input-image-preview" src={previewUrl} style={{height: "100%"}}
                             alt="Selected file"/>
                    ) : (
                        <div className="cta-text">
                            <p>Upload an image</p>
                            <p>Drag and drop your <i className="fa fa-solid fa-image"/></p>
                        </div>
                    )}
                </div>
            </label>
            {dragActive && (
                <div
                    id="drag-file-element"
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                />
            )}
        </form>
    );
}

export default InputImage;