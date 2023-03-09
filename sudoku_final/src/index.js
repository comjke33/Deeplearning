import React from "react";
import ReactDOM from "react-dom";
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';
//css 파일 불러오기
import "./styles.css"
tf.setBackend('webgl')


async function load_model(){
    const model = await loadGraphModel("https://127.0.0.1:8080/model.json");
    return model;
}

let classesDir = {
    0: {
        name: '0'
        id: 0,
    },
    1: {
        name: '1'
        id: 1,
    },
    2: {
        name: '2'
        id: 2,
    },
    3: {
        name: '3'
        id: 3,
    },
    4: {
        name: '4'
        id: 4,
    },
    5: {
        name: '5'
        id: 5,
    },
    6: {
        name: '6'
        id: 6,
    },
    7: {
        name: '7'
        id: 7,
    },
    8: {
        name: '8'
        id: 8,
    },
    9: {
        name: '9'
        id: 9,
    }
}

