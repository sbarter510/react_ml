import React, { useState, useEffect, forwardRef } from 'react'
import Modal from '../common/Modal'
import axios from "axios";
import "./Launchpad.css"
import Button from "../common/Button"
var heatmap = require("./heatmap.png");
var distplot = require("./distplot.png")


const Launchpad = () => {

    const [userCsv, setUserCsv] = useState(false);
    const [formData, setFormData] = useState({});
    const [df, setDf] = useState();
    const [colNames, setColNames] = useState([]);
    const [targetVariable, setTargetVariable] = useState();
    const [submitSelectedTarget, setSubmitSelectedTarget] = useState(false);
    const [summary, setSummary] = useState();
    const [corr, setCorr] = useState();
    const [results, setResults] = useState(null);

    // TODO address e:any
    const onChangeHandler = (e: any) => {
        const file = e.target.files[0];
        const data = new FormData();
        data.append('file', file);
        data.append('filename', "watevs");
        setFormData(data);
        setUserCsv(true);
    }

    const getColNames = () => {
        const targetSelect = document.getElementById("target-select");
        if (colNames) {
            colNames.map((col, idx) => {
                let option = document.createElement("option")
                targetSelect?.appendChild(option).append(col)
                return null

            })
        }
    }

    useEffect(() => {
        getColNames()
    }, [colNames])


    useEffect(() => {
        const url = "http://localhost:5000/data"
        if (userCsv) {
            axios.post(url, formData, { headers: { 'Access-Control-Allow-Origin': '*', 'content-type': 'text/json' } }).then((response) => {
                console.log(response)
                document.getElementById("data-table")?.append(response.data)
                setDf(response.data.table)
                setColNames(response.data.col_names)
            })

        }
    }, [userCsv, formData])

    const setTable = () => {
        if (df) {
            return { __html: df }
        }
    }

    const displaySummary = () => {
        if (summary!) {
            return { __html: summary }
        }
    }

    const displayCorr = () => {
        if (corr!) {
            return { __html: corr }
        }
    }

    // Adds bootstrap class to pandas table and removes some alignment bug
    useEffect(() => {
        for (var i = 0; i < document.getElementsByClassName("dataframe").length; i++) {
            const table = document.getElementsByClassName("dataframe")[i];
            const head = document.querySelectorAll("thead")[i];
            const body = document.querySelectorAll("tbody")
            if (df && table) {
                table.classList.add("table")
                table.classList.add("table-dark")
                for (let i = 0; i < head.childNodes.length; i++) {
                    head.querySelectorAll("tr").forEach((row) => {
                        row.style.textAlign = '';
                    })

                }
            }
            summary && document.getElementById("corr")?.querySelectorAll('td').forEach((td) => {
                if (Number(td.innerText) > 0.5 || Number(td.innerText) < -0.5) {
                    td.style.color = "green"
                }
            })
        }
    }, [df, summary, corr])

    const targetOnChangeHandler = (e: any) => {
        setTargetVariable(e.target.value)
    }

    const onTargetSelectionHandler = () => {
        // make a BE request which identifies the type of column which we then decide whether its classification or regression
        setSubmitSelectedTarget(true)
    }

    useEffect(() => {
        if (submitSelectedTarget) {
            axios.post(`http://localhost:5000/inspect/${targetVariable}`, { target: targetVariable }).then(response => {
                setSummary(response.data.summary)
                setCorr(response.data.corr)
            })
        }
        return () => {
            setSubmitSelectedTarget(false)
        }
    }, [submitSelectedTarget, targetVariable])

    const trainClickHandler = () => {
        // SEND REQEUST TRAIN MODEL RETURN TRAINING SCORE
        axios.get(`http://localhost:5000/train/${targetVariable}`).then(response => setResults(response.data))
    }

    const showResults: Function = (results: any) => {
        return <><h2>Train Score: {results!.training_score}</h2> <h2>Test Score: {results!.test_score}</h2></>
    }

    return (
        <>
            {/* Upload CSV */}
            {!df &&
                <Modal >
                    <h2>Upload a csv</h2>
                    {/* graphic */}
                    <input type="file" id="csv" onChange={onChangeHandler} />
                </Modal >
            }
            {/* Select Target Feature */}
            {colNames && getColNames()}
            {df && !summary && <Modal >
                <h2>Select your target variable</h2>
                <select id="target-select" onChange={(e: any) => targetOnChangeHandler(e)}>
                </select>
                <div className="table-container" dangerouslySetInnerHTML={setTable()} >

                </div>
                <Button label={`Predict ${targetVariable}`} onClickHandler={() => onTargetSelectionHandler()} />
            </Modal>}
            {/* Display Summary before training */}
            {summary && !results &&
                <>
                    <div className="row">
                        <div className="col col-2">
                            <div className="container">
                                <div id="summary" className="summary-container" dangerouslySetInnerHTML={displaySummary()}>
                                </div>
                            </div>
                        </div>

                        <div className="col col-5">
                            {/* <div className="container"> */}
                            <img src={heatmap} alt="ham" className="chart" />
                            {/* </div> */}
                        </div>

                        <div className="col col-5">
                            <div className="image-container">
                                <img src={distplot} alt="distribution plot" className="chart" />
                            </div>
                        </div>
                    </div>

                    <div className="row">
                        <div className="col">
                            <div id="corr" className="corr-container" dangerouslySetInnerHTML={displayCorr()}></div>
                        </div>
                    </div>
                    <Button label="train" onClickHandler={trainClickHandler} ></Button>
                </>}
            {/* Results */}
            {results && showResults(results)}

        </>
    )
}

export default Launchpad;   
