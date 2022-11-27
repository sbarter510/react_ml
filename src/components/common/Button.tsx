import React from 'react'

type ButtonProps = {
    label: string
    onClickHandler: Function
}

export default function Button(props: ButtonProps) {
    return (
        <button type="button" className="btn btn-primary" onClick={() => props.onClickHandler()} >{props.label}</button >
    )
}
