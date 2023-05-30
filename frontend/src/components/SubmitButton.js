import {Button} from "react-bootstrap";

const SubmitButton = ({loading, selectedImg, handleSubmit}) => {
    return (
        <div className="d-flex justify-content-center my-3">
            {loading ? (
                <div className="spinner-border text-primary" role="status">
                    <span className="sr-only">Loading...</span>
                </div>
            ) : (
                <Button
                    type="submit"
                    className="btn btn-primary btn-lg"
                    onClick={handleSubmit}
                    disabled={!selectedImg}
                >
                    Submit
                </Button>
            )}
        </div>
    );
};

export default SubmitButton;