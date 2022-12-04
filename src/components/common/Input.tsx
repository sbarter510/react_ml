import React, { useRef, useState } from 'react'

export default function Input(props: { inputLabel: string }) {
    const { inputLabel } = props;
    const [val, setVal] = useState("");
    const onChangeHandler = () => {
        setVal(ref.current!.value)
        // document.getElementById(inputLabel)!.value = val
        console.log(val)
    }
    const ref = useRef<null | HTMLInputElement>(null);
    return (
        <>
            <label htmlFor={inputLabel} >{inputLabel}</label>
            <input type="text" id={inputLabel} value={val} ref={ref} onChange={onChangeHandler}></input>
        </>
    )
}
