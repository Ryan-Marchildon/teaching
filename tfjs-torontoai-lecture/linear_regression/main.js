// ********************************
// Linear Regression in the Browser
// ********************************

// Link to TFJS documentation: https://js.tensorflow.org/api/1.0.0/ 


// Generate and plot some data
const SLOPE = 1.8;
const INTERCEPT = -0.3;
const data = makeRandomData(xmin=-0.2, xmax=1.4, numpoints=100);
plotData(data);


// ********
// (1). *** Convert data from Array into tensors ***
// ********



// ********
// (2). *** Define Model ***
// ********
// In this case, a simple model for linear regression
// (Dense layer with 1 weight and 1 bias, i.e. y = W*x + B)



// ********
// (3). *** Define the optimizer parameters and compile model ***
// ********



// ********
// (4). *** Train the model ***
// ********
document.getElementById("trainModel").onclick = function(){
    
    // add TFJS code here =============


    //document.getElementById('getRegLine').disabled = false;
    // ================================

}


// ********
// (5). *** Generate Predictions ***
// ********
document.getElementById("getRegLine").onclick = function(){

    // add TFJS code here =============

    
    // ================================

    // plotDataAndRegLine(data, linePreds);
    // document.getElementById('getPred').disabled = false;

}

document.getElementById("getPred").onclick = function(){

    // let xInput = parseFloat(document.getElementById("xInput").value);
    // console.log('User Submitted:', xInput);

    // add TFJS code here =============

    
    // ================================

    // document.getElementById('prediction').innerHTML = parseFloat(predVal.toFixed(3)); 

}





// *****************************
// Helper Code (Leave Unaltered)
// *****************************

// initial page element settings (get hoisted to the top of the script)
document.getElementById('getRegLine').disabled = true;
document.getElementById('getPred').disabled = true;


function makeRandomData(xmin, xmax, numpoints, sigma=1.5){

    const x = Array(numpoints).fill().map(() => 
        Math.random() * (xmax - xmin) + xmin
    );

    const y_noise = Array(numpoints).fill().map(() => 
        random_normal() * sigma - 0.5
    );

    const y = x.map(function(num, idx){
        return num*SLOPE + y_noise[idx] + INTERCEPT;
    });

    return {
        'x' : x,
        'y' : y
    }
}

// function for generating normally-distributed random numbers
// between 0 and 1 based on the Box-Muller transform
function random_normal() {
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    let num = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    num = num / 10.0 + 0.5; // Translate to 0 -> 1
    if (num > 1 || num < 0) return random_normal(); // resample between 0 and 1
    return num;
}


function plotData(dataObj){

    const layout = {title: "Data"}

    const dataTrace = {
        x: data['x'],
        y: data['y'],
        mode: 'markers',
        type: 'scatter',
        name: 'Raw Data'
    };

    let traces = [dataTrace];

    Plotly.newPlot('plotDiv', traces, layout); 
 
}


function plotDataAndRegLine(dataObj, regPreds){

    const layout = {title: "Data"}

    const dataTrace = {
        x: data['x'],
        y: data['y'],
        mode: 'markers',
        type: 'scatter',
        name: 'Raw Data'
    };

    const regLineTrace = {
        x: data['x'],
        y: regPreds,
        mode: 'line',
        type: 'scatter',
        name: 'Regression Line'
    };

    let traces = [dataTrace, regLineTrace];

    Plotly.newPlot('plotDiv', traces, layout); 
 
}


function plotLoss(modelHistory){

    const loss = modelHistory.history.loss;

    const layout = {
        title: "Training Loss",
        xaxis: {
            title: {
                text: 'Training Epoch'
            }
        },
        yaxis: {
            title: {
                text: 'Loss'
            }
        }
    }

    const lossTrace = {
        x: [...Array(loss.length).keys()],
        y: loss,
        mode: 'lines',
        type: 'scatter'
    };

    let traces = [lossTrace];

    Plotly.newPlot('lossDiv', traces, layout); 
 
}