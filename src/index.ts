import { loadFileAsArrayBuffer, PackedGaussians } from './ply';
import { CameraFileParser, InteractiveCamera } from './camera';
import { Renderer } from './renderer';
import { vec3 } from 'wgpu-matrix';

if (!navigator.gpu) {
    alert("WebGPU not supported on this browser! (navigator.gpu is null)");
}

// grab the DOM elements
const canvas = document.getElementById("canvas-webgpu") as HTMLCanvasElement;
const loadingPopup = document.getElementById('loading-popup')! as HTMLDivElement;
const fpsCounter = document.getElementById('fps-counter')! as HTMLLabelElement;
const cameraFileInput = document.getElementById('cameraButton')! as HTMLInputElement;
const cameraList = document.getElementById('cameraList')! as HTMLUListElement;
const plyFileInput = document.getElementById('plyButton') as HTMLInputElement;


canvas.width = window.innerWidth;
canvas.height = window.innerHeight;


// create the camera and renderer globals
let interactiveCamera = InteractiveCamera.default(canvas);
var currentRenderer: Renderer;
var gaussians: PackedGaussians;

// swap the renderer when the ply file changes
function handlePlyChange(event: any) {
    const file = event.target.files[0];

    async function onFileLoad(arrayBuffer: ArrayBuffer) {
        if (currentRenderer) {
            await currentRenderer.destroy();
        }
        gaussians = new PackedGaussians(arrayBuffer);
        try {
            const context = await Renderer.requestContext(gaussians);
            const renderer = new Renderer(canvas, interactiveCamera, gaussians, context, fpsCounter);
            currentRenderer = renderer; // bind to the global scope

            loadingPopup.style.display = 'none'; // hide loading popup
        } catch (error) {
            loadingPopup.style.display = 'none'; // hide loading popup
            alert(error);
        }
    }

    if (file) {
        loadingPopup.style.display = 'block'; // show loading popup
        loadFileAsArrayBuffer(file)
            .then(onFileLoad);
    }
}

// loads the default ply file (bundled with the source) at startup, useful for dev
async function loadDefaultPly() {
    // Parse url
    console.log(window.location.href);
    const urlObj = new URL(window.location.href);
    const params = new URLSearchParams(urlObj.search);
    var url = "ply.ply";
    if(params.has("model")){
        const model_name = params.get('model');
        url = model_name+".ply";
    }

    // Get specific parameter values
    //const url = "ply.ply";
    loadingPopup.style.display = 'block'; // show loading popup
    const content = await fetch(url);
    const arrayBuffer = await content.arrayBuffer();
    gaussians = new PackedGaussians(arrayBuffer);
    const context = await Renderer.requestContext(gaussians);
    const renderer = new Renderer(canvas, interactiveCamera, gaussians, context, fpsCounter);
    currentRenderer = renderer; // bind to the global scope
    loadingPopup.style.display = 'none'; // hide loading popup
   
}


// DEV: uncomment this line to load the default ply file at startup
loadDefaultPly().then(()=>{
    let center = vec3.scale( vec3.add( gaussians.min_pos , gaussians.max_pos) , 0.5);
    interactiveCamera.setCenter(center);
    
});

// add event listeners
plyFileInput!.addEventListener('change', handlePlyChange);
const camParser = new CameraFileParser(
    cameraFileInput,
    cameraList,
    canvas,
    (camera) => interactiveCamera.setNewCamera(camera),
);



async function loadDefaultCamera(){
    const url = "cam.json";
    const content = await fetch(url);
    if (content.ok) {
        const data = await content.json();        
        camParser.handleJsonData(data);
        //interactiveCamera.setNewCamera(camParser.cameraList[0]);
        //console.log("current camera " + interactiveCamera.getCamera().viewMatrix);
        //console.log("Json camera " + camParser.cameraList[0].viewMatrix);
    } 
}
loadDefaultCamera();

// Resize callback
function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    interactiveCamera.resize();
    currentRenderer.resize();
}

window.addEventListener('resize', resizeCanvas);


