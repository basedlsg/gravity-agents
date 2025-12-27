/**
 * PhysicsWorld - Wrapper around cannon-es physics engine
 */

import * as CANNON from 'cannon-es';

export class PhysicsWorld {
  constructor(options = {}) {
    this.gravity = options.gravity ?? 9.81;
    this.timeStep = options.timeStep ?? 1 / 60;

    // Create cannon-es world
    this.world = new CANNON.World();
    this.world.gravity.set(0, -this.gravity, 0);

    // Solver settings for stability
    this.world.solver.iterations = 10;
    this.world.allowSleep = false;

    // Default contact material - low friction for responsive movement
    this.defaultMaterial = new CANNON.Material('default');
    this.world.defaultContactMaterial = new CANNON.ContactMaterial(
      this.defaultMaterial,
      this.defaultMaterial,
      {
        friction: 0.01,  // Very low friction for game-like movement
        restitution: 0.1
      }
    );

    // Track bodies
    this.bodies = new Map();
  }

  setGravity(g) {
    this.gravity = g;
    this.world.gravity.set(0, -g, 0);
  }

  step() {
    this.world.step(this.timeStep);
  }

  addBody(name, body) {
    body.material = this.defaultMaterial;
    this.world.addBody(body);
    this.bodies.set(name, body);
    return body;
  }

  removeBody(name) {
    const body = this.bodies.get(name);
    if (body) {
      this.world.removeBody(body);
      this.bodies.delete(name);
    }
  }

  getBody(name) {
    return this.bodies.get(name);
  }

  createBox(name, options) {
    const {
      position = [0, 0, 0],
      size = [1, 1, 1],
      mass = 0,
      type = mass === 0 ? CANNON.Body.STATIC : CANNON.Body.DYNAMIC
    } = options;

    const halfExtents = new CANNON.Vec3(size[0] / 2, size[1] / 2, size[2] / 2);
    const shape = new CANNON.Box(halfExtents);
    const body = new CANNON.Body({
      mass,
      type,
      position: new CANNON.Vec3(...position),
      shape
    });

    return this.addBody(name, body);
  }

  createCapsule(name, options) {
    const {
      position = [0, 0, 0],
      radius = 0.4,
      height = 1.8,
      mass = 70
    } = options;

    // Approximate capsule with cylinder + spheres
    const cylinderHeight = height - 2 * radius;
    const cylinderShape = new CANNON.Cylinder(radius, radius, cylinderHeight, 8);
    const sphereShape = new CANNON.Sphere(radius);

    const body = new CANNON.Body({
      mass,
      position: new CANNON.Vec3(...position),
      fixedRotation: true // Prevent tumbling
    });

    // Add shapes
    body.addShape(cylinderShape, new CANNON.Vec3(0, 0, 0));
    body.addShape(sphereShape, new CANNON.Vec3(0, cylinderHeight / 2, 0));
    body.addShape(sphereShape, new CANNON.Vec3(0, -cylinderHeight / 2, 0));

    return this.addBody(name, body);
  }

  createSphere(name, options) {
    const {
      position = [0, 0, 0],
      radius = 0.5,
      mass = 1
    } = options;

    const shape = new CANNON.Sphere(radius);
    const body = new CANNON.Body({
      mass,
      position: new CANNON.Vec3(...position),
      shape
    });

    return this.addBody(name, body);
  }

  reset() {
    // Remove all bodies
    for (const name of this.bodies.keys()) {
      this.removeBody(name);
    }
  }

  getBodyState(name) {
    const body = this.bodies.get(name);
    if (!body) return null;

    return {
      position: [body.position.x, body.position.y, body.position.z],
      velocity: [body.velocity.x, body.velocity.y, body.velocity.z],
      quaternion: [body.quaternion.x, body.quaternion.y, body.quaternion.z, body.quaternion.w]
    };
  }
}
