import React from "react";
import {LANGUAGES} from "../utils/constants";

const SelectLanguage = ({language, handleSelectLanguage}) => {
    return (
        <div className="text-center">
            <select
                className="custom-select mt-3 text-capitalize"
                value={language}
                onChange={e => {
                    handleSelectLanguage(e.target.value)
                }}
            >
                {LANGUAGES.map(lang => <option value={lang} className="text-capitalize" key={lang}>{lang}</option>)}

            </select>
        </div>
    );
};

export default SelectLanguage;