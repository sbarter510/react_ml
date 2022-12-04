import React, { useState } from 'react'
import Input from './common/Input'
import "./PredictInputs.css"
import axios from "axios"


export default function PredictInputs(props: { inputs: string[] }) {
    const { inputs } = props;
    const [result, setResult] = useState("");
    const displayInputs = () => {
        return inputs.map((input) => <Input inputLabel={input} />)
    }


    const onClickHandler = () => {
        // collect the input values
        const userInputArray = inputs.map(input => {
            return document.getElementById(input)!.getAttribute("value")
        })
        console.log(userInputArray)

        // make BE request passing data
        axios.post("http://localhost:5000/predict", { userInputArray: userInputArray }).then(response => {
            console.log(response)
            setResult(response.data[0])
        })
        return null;
    }



    return <div className="prediction-inputs-container"><h2 >Enter your own data below to make a prediction.</h2>{displayInputs()}<button type="button" className="btn btn-primary" onClick={() => onClickHandler()}>Make Prediction</button></div>
}
