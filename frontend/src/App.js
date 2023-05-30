import React, {useState} from "react";
import {ToastContainer} from "react-toastify";
import {Container, Row, Col} from "react-bootstrap";

import Header from "./components/Header";
import InputImage from "./components/InputImage";
import SelectLanguage from "./components/SelectLanguage";
import SubmitButton from "./components/SubmitButton";
import Footer from "./components/Footer";
import {submitData} from "./api/api";
import {LANGUAGES} from "./utils/constants";
import DisplayHeatmap from "./components/DisplayHeatmap";


const App = () => {
    const [selectedImage, setSelectedImage] = useState(null);
    const [language, setLanguage] = useState(LANGUAGES[0]);
    const [loading, setLoading] = useState(false);
    const [response, setResponse] = useState(null);

    const handleSelectImage = (image) => {
        // Clear previous results
        setResponse(null);
        setSelectedImage(image);
    }

    const handleSelectLanguage = (lang) => {
        // Clear previous results
        setResponse(null);
        setLanguage(lang);
    }

    const handleSubmit = async () => {
        setLoading(true);

        const formData = new FormData();
        formData.append('image', selectedImage);
        formData.append('language', language);
        const result = await submitData(formData);
        if (result) {
            console.log(result);
            setResponse(result);
        }

        setLoading(false);
    };


    return (
        <div className="App">
            <Container>
                <Row className="justify-content-center">
                    <Col xs={12} md={8} lg={6}>
                        <Header/>
                        <InputImage handleSelectImage={handleSelectImage}/>
                        <SelectLanguage language={language} handleSelectLanguage={handleSelectLanguage}/>

                        {!response &&
                            <SubmitButton loading={loading} selectedImg={selectedImage} handleSubmit={handleSubmit}/>}

                        {response &&
                            <div className="result mt-5">
                                <h3 className="text-center">Caption</h3>
                                <p className="caption text-center">{response["text"].toUpperCase()}</p>
                            </div>}
                    </Col>
                </Row>
            </Container>

            {response && <DisplayHeatmap base64String={response["heatmap_base64"]}/>}

            <Footer/>

            <ToastContainer
                position="bottom-right"
                autoClose={1500}
                closeOnClick
                rtl={false}
                hideProgressBar={true}
                pauseOnFocusLoss
                draggable
                pauseOnHover
            />
        </div>
    );
}


export default App;
