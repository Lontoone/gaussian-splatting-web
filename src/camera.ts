import { mat4, vec3, mat3, Mat4, Mat3, Vec3, vec4 } from 'wgpu-matrix';


const FOV = 1.04719755; // 60 degree

// camera as loaded from JSON
interface CameraRaw {
    id: number;
    img_name: string;
    width: number;
    height: number;
    position: number[];
    rotation: number[][];
    fx: number;
    fy: number;
}

// for some reason this needs to be a bit different than the one in wgpu-matrix
function getProjectionMatrix(znear: number, zfar: number, fovX: number, fovY: number): Mat4 {
    const tanHalfFovY: number = Math.tan(fovY / 2);
    const tanHalfFovX: number = Math.tan(fovX / 2);

    const top: number = tanHalfFovY * znear;
    const bottom: number = -top;
    const right: number = tanHalfFovX * znear;
    const left: number = -right;

    const P: Mat4 = mat4.create();

    const z_sign: number = 1.0;

    P[0] = (2.0 * znear) / (right - left);
    P[5] = (2.0 * znear) / (top - bottom);
    P[8] = (right + left) / (right - left);
    P[9] = (top + bottom) / (top - bottom);
    P[10] = z_sign * zfar / (zfar - znear);
    P[11] = -(zfar * znear) / (zfar - znear);
    P[14] = z_sign;
    P[15] = 0.0;

    return mat4.transpose(P);
}

// useful for coordinate flips
function diagonal4x4(x: number, y: number, z: number, w: number): Mat4 {
    const m = mat4.create();
    m[0] = x;
    m[5] = y;
    m[10] = z;
    m[15] = w;
    return m;
}

// A camera as used by the renderer. Interactivity is handled by InteractiveCamera.
export class Camera {
    height: number;
    width: number;
    viewMatrix: Mat4;
    perspective: Mat4;
    focalX: number;
    focalY: number;
    scaleModifier: number;

    m_up:number = 1.0;
    center : Vec3 = [0,0,0];
    
    radius = 3.0;
    _eye : Vec3 = [0,-5,3];
    previous_eye : Vec3 =[0,-5,3];
    last_dir : Vec3 = vec3.create();
    inertia :number = 0 ;

    sensity_slider:HTMLInputElement;
    get_sensity = (): number => {
        return parseFloat(this.sensity_slider.value)/1000;
    };

    constructor(
        height: number,
        width: number,
        viewMatrix: Mat4,
        perspective: Mat4,
        focalX: number,
        focalY: number,
        scaleModifier: number,
    ) {
        this.height = height;
        this.width = width;
        this.viewMatrix = viewMatrix;
        this.perspective = perspective;

        this.focalX = focalX;
        this.focalY = focalY;
        this.scaleModifier = scaleModifier;
        this.sensity_slider = document.getElementById('sensity_slider') as HTMLInputElement;



    }

    static default(): Camera {
        return new Camera(
            window.innerHeight,
            window.innerWidth,
            mat4.lookAt([0, 0 , -5], [0, 0, 0], [0, 1, 0]),            
            mat4.perspective(FOV, innerWidth/innerHeight, 0.03, 1000),
            window.innerWidth,
            window.innerHeight,
            1,
        )
    }
    update(){       
        if(this.inertia==0){                        
            this.viewMatrix = mat4.lookAt( this._eye, this.center , [0,1,0]);
        }
        else{
            let lst_dir = vec3.subtract(this._eye , this.previous_eye);
            vec3.normalize(lst_dir,lst_dir);                    
            
            this._eye = vec3.add(this._eye , vec3.scale(lst_dir , 0.05));
            this.viewMatrix = mat4.lookAt( this._eye, this.center , [0,1,0]);
            this.inertia-=0.05;

        }
    }
    // computes the depth of a point in camera space, for sorting
    dotZ(): (v: Vec3) => number {
        const depthAxis = this.depthAxis();
        return (v: Vec3) => {
            return vec3.dot(depthAxis, v);
        }
    }

    // gets the camera position in world space, for evaluating the spherical harmonics
    getPosition(): Vec3 {
        const inverseViewMatrix = mat4.inverse(this.viewMatrix);
        return mat4.getTranslation(inverseViewMatrix);
    }

    getProjMatrix(): Mat4 {
        var flippedY = mat4.clone(this.perspective);
        flippedY = mat4.mul(flippedY, diagonal4x4(1, -1, 1, 1));
        return mat4.multiply(flippedY, this.viewMatrix);
    }

    // for camera interactions
    translate(x: number, y: number, z: number) {      
        x   *=-1;
        y   *=-1;
        const world_up = vec3.create(0,1,0);

        let fz = vec3.subtract(this._eye , this.center);
        let len = vec3.length(fz)/0.785; //45 degrees
        fz = vec3.normalize(fz);
        let fx = vec3.cross(world_up , fz);
        let fy = vec3.cross(fz , fx);
        fx = vec3.normalize(fx);
        fy = vec3.normalize(fy);

        let panVector = vec3.scale( vec3.add( vec3.scale(fx , -x) ,vec3.scale(fy , y))  , len);
        vec3.add( panVector , this._eye , this._eye);
        vec3.add( panVector , this.center , this.center);

    }    
    zoom(distance: number){
        
        this.radius = this.radius + distance * 0.001;
        this.radius = Math.max(this.radius , 0.2);          
        
        let origin = this.center;
        let postion = this._eye;
        let centerToEye = vec3.subtract(postion, origin);
        centerToEye = vec3.normalize(centerToEye);
        vec3.scale(centerToEye , this.radius , centerToEye);

        // Finding the new position
        let newPosition = vec3.create();
        vec3.add(centerToEye , origin , newPosition);

        this._eye = newPosition;
    }

    // for camera interactions
    rotate(x: number, y: number, z: number) {
        

        if(x == 0 && y == 0)
            return;
        

        const two_pi = Math.PI * 2;
        x*=two_pi * this.get_sensity();
        y*=two_pi * this.get_sensity();

        let origin = this.center;
        let postion = this._eye;

        let centerToEye = vec3.subtract(postion, origin);
        //let radius = vec3.length(centerToEye);
        let radius = this.radius;
        centerToEye = vec3.normalize(centerToEye);
        let axe_z = centerToEye;

        // Find the rotation around the UP axis (Y)
        const world_up = vec3.create(0,1,0);
        let rot_y = mat4.rotate(mat4.identity() , world_up , -x );
        // Apply the (Y) rotation to the eye-center vector
        centerToEye = mat4.multiply( rot_y , vec4.create(centerToEye[0] , centerToEye[1],centerToEye[2] , 0));

        // Find the rotation around the X vector: cross between eye-center and up (X)
        let axe_x = vec3.normalize( vec3.cross(world_up , axe_z));
        let rot_x = mat4.rotate(mat4.identity() , axe_x ,-y);

        // Apply the (X) rotation to the eye-center vector
        let vect_rot = mat4.mul(rot_x , vec4.create(centerToEye[0],centerToEye[1],centerToEye[2],0  ));

        //Avoid flipping
        if(
            (Math.sign(vect_rot[0] ) == Math.sign(centerToEye[0])) &&
            (Math.abs(vect_rot[1] -centerToEye[1])<0.2) 
        ){
            centerToEye = vect_rot;
        }
        
        // Make the vector as long as it was originally
        vec3.scale(centerToEye , radius , centerToEye);

        // Finding the new position
        let newPosition = vec3.create();
        vec3.add(centerToEye , origin , newPosition);

        this._eye = newPosition;   
        
    }

    // the depth axis is the third column of the transposed view matrix
    private depthAxis(): Vec3 {
        return mat4.getAxis(mat4.transpose(this.viewMatrix), 2);
    }
}

// Adds interactivity to a camera. The camera is modified by the user's mouse and keyboard input.
export class InteractiveCamera {
    resize() {
        this.camera.perspective = mat4.perspective(FOV, innerWidth/innerHeight, 0.03, 1000);
    }
    private camera: Camera;
    private canvas: HTMLCanvasElement;

    private mode = 0;
    private drag: boolean = false;
    private oldX: number = 0;
    private oldY: number = 0;
    private dRX: number = 0;
    private dRY: number = 0;
    private dRZ: number = 0;
    private dTX: number = 0;
    private dTY: number = 0;
    private dTZ: number = 0;

    private dirty: boolean = true;

    constructor(camera: Camera, canvas: HTMLCanvasElement) {
        this.camera = camera;
        this.canvas = canvas;

        this.createCallbacks();
    }

    static default(canvas: HTMLCanvasElement): InteractiveCamera {
        return new InteractiveCamera(Camera.default(), canvas);
    }

    private createCallbacks() {
        document.addEventListener('contextmenu', (event) => {
            event.preventDefault();
        });

        //===============================
        //      Finger
        //===============================
        document.addEventListener('touchstart', (event) => {
            if(event.touches.length ==1){
                this.mode = 2; // Rotate
            }
            if(event.touches.length == 2){
                this.mode = 0; // pan
            }
            this.drag = true;
            this.oldX = event.touches[0].clientX;
            this.oldY = event.touches[0].clientY;
            this.setDirty();                
            event.preventDefault();

        });
        
        document.addEventListener('touchmove', (event) => {
            if (!this.drag) return false;
            if(this.mode==2){                
                this.dRX = (event.touches[0].clientX - this.oldX) * 2 * Math.PI / this.canvas.width;
                this.dRY = -(event.touches[0].clientY - this.oldY) * 2 * Math.PI / this.canvas.height;
                
                this.oldX = event.touches[0].clientX;
                this.oldY = event.touches[0].clientY;
                this.setDirty();
                event.preventDefault();
            }

            else if(this.mode==0){
                console.log("Mouse btn 0");
                this.dTX = (event.touches[0].clientX - this.oldX) * 2  / this.canvas.width;
                this.dTY = -(event.touches[0].clientY - this.oldY) * 2 / this.canvas.height;
                
                this.oldX = event.touches[0].clientX;
                this.oldY = event.touches[0].clientY;
                this.setDirty();
                event.preventDefault();

            }
        });
        
        document.addEventListener('touchend', (event) => {
            console.log('Touch end detected');
            this.drag = false;            
            this.mode = -1;
        });

        //===============================
        //      Kyeboard Input
        //===============================
        this.canvas.addEventListener('mousedown', (e) => {
            this.mode = e.button;
            
            // Right button for moving look center
            if (e.button == 2) {                
                this.drag = true;
                this.oldX = e.pageX;
                this.oldY = e.pageY;
                this.setDirty();                
                e.preventDefault();
            }
            // Left button for rotating view
            if(e.button ==0){
                this.drag = true;
                this.oldX = e.pageX;
                this.oldY = e.pageY;
                this.setDirty();                
            }
        }, false);
        document.addEventListener('wheel', (event) => {
            this.camera.zoom(event.deltaY);
            
            this.setDirty();
        });

        this.canvas.addEventListener('mouseup', (e) => {            
            this.drag = false;            
            this.mode = -1;

            this.camera.inertia =  1;                        
            this.camera.previous_eye = this.camera._eye;            
            
        }, false);

        this.canvas.addEventListener('mousemove', (e) => {            
            if (!this.drag) return false;            
            this.camera.inertia =  0; 

            if(this.mode==2){                
                this.dRX = (e.pageX - this.oldX) * 2 * Math.PI / this.canvas.width;
                this.dRY = -(e.pageY - this.oldY) * 2 * Math.PI / this.canvas.height;
                
                this.oldX = e.pageX;
                this.oldY = e.pageY;
                this.setDirty();
                e.preventDefault();
            }

            else if(this.mode==0){
                console.log("Mouse btn 0");
                this.dTX = (e.pageX - this.oldX) * 2  / this.canvas.width;
                this.dTY = -(e.pageY - this.oldY) * 2 / this.canvas.height;
                
                this.oldX = e.pageX;
                this.oldY = e.pageY;
                this.setDirty();
                e.preventDefault();

            }
        }, false);

        this.canvas.addEventListener('wheel', (e) => {
            this.dTZ = e.deltaY * 0.1;
            this.setDirty();
            e.preventDefault();
        }, false);

        window.addEventListener('keydown', (e) => {
            const keyMap: {[key: string]: () => void} = {
                // translation                
                //'a': () => { this.dTX -= 0.1 },
                //'d': () => { this.dTX += 0.1 },
                
                'q': () => { this.dTZ += 0.1 },
                'e': () => { this.dTZ -= 0.1 },

                // rotation
                'j': () => { this.dRX += 0.1 },
                'l': () => { this.dRX -= 0.1 },
                'i': () => { this.dRY += 0.1 },
                'k': () => { this.dRY -= 0.1 },
                'u': () => { this.dRZ += 0.1 },
                'o': () => { this.dRZ -= 0.1 },
            }

            if (!keyMap[e.key]) {
                return;
            } else {
                keyMap[e.key]();
                this.setDirty();
                e.preventDefault();
            }

        }, false);
    }

    public setNewCamera(newCamera: Camera) {        
        this.camera = newCamera;        
        this.setClean();
    }

    public setDirty() {
        this.dirty = true;
    }
    
    private setClean() {
        this.dirty = false;
    }

    public isDirty(): boolean {
        return this.camera.inertia >0 || this.dirty;
    }

    public getCamera(): Camera {
        if (this.isDirty()) {                       

            this.camera.rotate(this.dRX, this.dRY, this.dRZ);
            this.camera.translate(this.dTX, this.dTY, this.dTZ);
            this.dTX = this.dTY = this.dTZ = this.dRX = this.dRY = this.dRZ = 0;

            this.camera.update();
            this.setClean();
            
        }

        return this.camera;
    }
    public setCenter(v:Vec3){
        this.camera.center = v;
    }
}

function focal2fov(focal: number, pixels: number): number {
    return 2 * Math.atan(pixels / (2 * focal));
}

function worldToCamFromRT(R: Mat3, t: Vec3): Mat4 {
    const R_ = R;
    const camToWorld = mat4.fromMat3(R_);
    const minusT = vec3.mulScalar(t, -1);
    mat4.translate(camToWorld, minusT, camToWorld);
    return camToWorld;
}

// converting camera coordinate systems is always black magic :(
function cameraFromJSON(rawCamera: CameraRaw, canvasW: number, canvasH: number): Camera {
    //const fovX = focal2fov(rawCamera.fx, rawCamera.width);
    //const fovY = focal2fov(rawCamera.fy, rawCamera.height);
    //const asp = canvasW /  canvasH;    
    //const fovX = focal2fov(rawCamera.fx, window.innerWidth);
    //const fovY = focal2fov(rawCamera.fy, window.innerHeight);
    const fovX = focal2fov(rawCamera.fx , canvasW);
    const fovY = focal2fov(rawCamera.fy, canvasH);
    const projectionMatrix = getProjectionMatrix(0.2, 100, fovX, fovY);
    //const projectionMatrix = mat4.perspective(1.04719755, 1, 0.03, 10000);
    //const projectionMatrix = mat4.perspective(1.04719755, window.innerWidth/window.innerHeight, 0.03, 10000);


    const R = mat3.create(...rawCamera.rotation.flat());
    const T = rawCamera.position;

    const viewMatrix = worldToCamFromRT(R, T);        
    return new Camera(
        canvasH,
        canvasW,
        viewMatrix,
        projectionMatrix,
        canvasH,//rawCamera.fx,
        canvasW,//rawCamera.fy,
        1,    
    );
    
}

// A UI component that parses a JSON file containing a list of cameras and displays them as a list,
// allowing the user to choose from presets.
export class CameraFileParser {
    private fileInput: HTMLInputElement;
    private listElement: HTMLUListElement;
    private currentLineId: number = 0;
    private canvas: HTMLCanvasElement;
    private cameraSetCallback: (camera: Camera) => void;
    public cameraList : Camera[] = [];

    constructor(
        fileInput: HTMLInputElement,
        listElement: HTMLUListElement,
        canvas: HTMLCanvasElement,
        cameraSetCallback: (camera: Camera) => void,
    ) {
        this.fileInput = fileInput;
        this.listElement = listElement;
        this.canvas = canvas;
        this.cameraSetCallback = cameraSetCallback;

        this.fileInput.addEventListener('change', this.handleFileInputChange);       
    }

    private handleFileInputChange = (event: Event) => {
        const file = this.fileInput.files?.[0];

        if (file) {
            const reader = new FileReader();
            reader.onload = this.handleFileLoad;
            reader.readAsText(file);
        }
    };

    public handleJsonData(json : Array<String>){
        //const jsonData = JSON.parse(json);
        json.forEach((cameraJSON: any) => {
            this.currentLineId++;
            const listItem = document.createElement('li');
            const camera = cameraFromJSON(cameraJSON, this.canvas.width, this.canvas.height);
            this.cameraList.push(camera);
            listItem.textContent = cameraJSON.img_name;
            listItem.addEventListener('click', this.createCallbackForLine(camera));
            this.listElement.appendChild(listItem);
        });
    }

    private handleFileLoad = (event: ProgressEvent<FileReader>) => {
        if (!event.target) return;

        const contents = event.target.result as string;
        const jsonData = JSON.parse(contents);

        this.currentLineId = 0;
        this.listElement.innerHTML = '';
        this.handleJsonData(jsonData);
        /*
        jsonData.forEach((cameraJSON: any) => {
            this.currentLineId++;
            const listItem = document.createElement('li');
            const camera = cameraFromJSON(cameraJSON, this.canvas.width, this.canvas.height);
            listItem.textContent = cameraJSON.img_name;
            listItem.addEventListener('click', this.createCallbackForLine(camera));
            this.listElement.appendChild(listItem);
        });
        */
    };

    private createCallbackForLine = (camera: Camera) => {
        return () => {
            this.cameraSetCallback(camera);
        };
    };
}