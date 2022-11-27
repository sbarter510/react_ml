import React from 'react'
import "./Modal.css"

type ModalProps = {
    children?: React.ReactNode;
}

const Modal: React.FC<ModalProps> = (props) => {
    return (
        <div className="modal-container">
            {props.children}
        </div>
    )
}

export default Modal