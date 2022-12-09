let form = document.getElementById("corefForm");
let canSubmit = true;

function handleForm(event) {
    event.preventDefault();
    if (canSubmit) {
        processText();
    }
}

form.addEventListener('submit', handleForm);

function processText() {
    document.getElementById("corefResult").innerHTML = "Processing...";
    sendRequest()
}

let inputText = "";
let corefArray = [];
let highlightData = [];
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext('2d');

function findCoreference(id) {
    return corefArray.find(element => element.id == id);
}

function addPronounsHighlight() {
    corefArray.forEach(element => {
        let id = element.id;
        let pronoun = element.pronoun_with_offset.word;
        let pronounOffset = element.pronoun_with_offset.offset;
        highlightData.push([id, pronoun, pronounOffset, "highlight", true]);
    });
}

function compareHighlightData(a, b) {
    if (a[2] > b[2]) {
        return -1;
    }
    if (a[2] < b[2]) {
        return 1;
    }
    return 0;
}

function highlight() {
    highlightData.sort(compareHighlightData);
    let resultText = inputText;
    highlightData.forEach(element => {
        let id = element[0];
        let word = element[1];
        let wordOffset = element[2];
        let highlightClass = element[3];
        let addOnClick = element[4];
        if (addOnClick) {
            resultText = resultText.substring(0,wordOffset) + `<span id="${id}" class='${highlightClass}' onclick='showCoref("${id}")'>` + resultText.substring(wordOffset,wordOffset+word.length) + "</span>" + resultText.substring(wordOffset + word.length);
        } else {
            resultText = resultText.substring(0,wordOffset) + `<span id="nounSpan" class='${highlightClass}'>` + resultText.substring(wordOffset,wordOffset+word.length) + "</span>" + resultText.substring(wordOffset + word.length);
        }
    });
    document.getElementById("corefResult").innerHTML = resultText;
    highlightData = [];
}

const sendRequest = async () => {
    canSubmit = false;
    document.getElementById("submitButton").disabled = !canSubmit;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    inputText = document.getElementById('corefText').value;
    const response = await fetch('http://localhost:5010/coreference', {
        method: 'POST',
        body: `{ "coreftext" : "${inputText}" }`,
        headers: {
            'Content-Type' : 'application/json'
        }
    }).catch((_error) => {
        document.getElementById("corefResult").innerHTML = "Connection error!";
    });
    if (response && response.status == 200) {
        const myJson = await response.json();
        corefArray = myJson.coreference;

        addPronounsHighlight();
        highlight();
    } else if (response && response.status != 200) {
        document.getElementById("corefResult").innerHTML = `Error ${response.status}, please try again!`;
    }
    canSubmit = true;
    document.getElementById("submitButton").disabled = !canSubmit;
}

function calculateHalfpoint(startLeft, endLeft) {
    let midLeft = 0;
    if (startLeft > endLeft) {
        midLeft = endLeft + ((startLeft-endLeft)/2);
    } else {
        midLeft = startLeft + ((endLeft-startLeft)/2);
    }
    return midLeft;
}

function showCoref(id) {
    let coref = findCoreference(id);
    let score = coref.score;
    let noun = coref.noun_with_offset.word;
    let nounOffset = coref.noun_with_offset.offset;
    addPronounsHighlight();
    highlightData.push([id, noun, nounOffset, "highlight-blue", false]);
    highlight();
    let element = document.getElementById(id);
    let pronounBoundingRect = element.getBoundingClientRect();
    let nounBoundingRect = document.getElementById("nounSpan").getBoundingClientRect();
    let startBottom = 0;
    let startLeft = calculateHalfpoint(pronounBoundingRect.left, pronounBoundingRect.right)-8;
    let endBottom = 0;
    let endLeft = calculateHalfpoint(nounBoundingRect.left, nounBoundingRect.right)-8;
    let midBottom = 0;
    let midLeft = calculateHalfpoint(startLeft, endLeft);
    if (startLeft > endLeft) {
        canvas.width = startLeft+10;
        midBottom = startBottom + (startLeft-endLeft)/4;
    } else {
        canvas.width = endLeft+10;
        midBottom = startBottom + (endLeft-startLeft)/4;
    }

    ctx.font = "15px Times New Roman";
    let metrics = ctx.measureText(score);
    let text_width = metrics.width;
    let text_height = metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent;
    let midPoint = getQuadraticCurvePoint(startLeft, 0, midLeft, midBottom, endLeft, endBottom, 0.5);
    canvas.height = midPoint.y+text_height+10+30;
    
    ctx.beginPath();
    ctx.moveTo(startLeft, 0);
    ctx.quadraticCurveTo(midLeft, midBottom, endLeft, endBottom);
    ctx.stroke();
    
    ctx.font = "15px Times New Roman";
    ctx.fillText(score, midPoint.x-(text_width/2), midPoint.y+text_height+5);
}

function _getQBezierValue(t, p1, p2, p3) {
    let iT = 1 - t;
    return iT * iT * p1 + 2 * iT * t * p2 + t * t * p3;
}

function getQuadraticCurvePoint(startX, startY, cpX, cpY, endX, endY, position) {
    return {
        x:  _getQBezierValue(position, startX, cpX, endX),
        y:  _getQBezierValue(position, startY, cpY, endY)
    };
}