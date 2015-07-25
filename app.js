var net = new convnetjs.Net
net.makeLayers([
    {type: 'input', out_sx: 1, out_sy: 1, out_depth: 2},
    {type: 'fc', num_neurons: 20, activation: 'relu'},
    {type: 'fc', num_neurons: 20, activation: 'relu'},
    {type: 'fc', num_neurons: 20, activation: 'relu'},
    {type: 'fc', num_neurons: 20, activation: 'relu'},
    {type: 'fc', num_neurons: 20, activation: 'relu'},
    {type: 'fc', num_neurons: 20, activation: 'relu'},
    {type: 'fc', num_neurons: 20, activation: 'relu'},
    {type: 'regression', num_neurons: 3}
])

var trainer = new convnetjs.SGDTrainer(net, {
    learning_rate: 0.01,
    momentum: 0.9,
    batch_size: 10,
    l2_decay: 0
})

var size = 250

var in0 = document.getElementById('in0')
in0.width = in0.height = size
var cin0 = in0.getContext('2d')

var out = document.getElementById('out')
out.width = out.height = size
var cout = out.getContext('2d')

function init() {
    cin0.fillStyle = '#fffbeb'
    cin0.fillRect(0, 0, size, size)

    cin0.font = '127px Nitti WM2'
    cin0.textAlign = 'center'
    cin0.textBaseline = 'middle'
    cin0.fillStyle = '#20201f'
    cin0.fillText('A', 0.5 * size, 0.5 * size)
}

function paint() {
    var p = cout.getImageData(0, 0, size, size)
    var vol = new convnetjs.Vol(1, 1, 2)

    for (var x = 0; x < size; ++x) {
        vol.w[0] = x / size - 0.5

        for (var y = 0; y < size; ++y) {
            vol.w[1] = y / size - 0.5

            var ip = 4 * (x + size * y)
            var r = net.forward(vol)

            p.data[ip] = 0|255 * r.w[0]
            p.data[ip + 1] = 0|255 * r.w[1]
            p.data[ip + 2] = 0|255 * r.w[2]
            p.data[ip + 3] = 255
        }
    }

    cout.putImageData(p, 0, 0)
}

function learn() {
    var p = cin0.getImageData(0, 0, size, size)
    var vol = new convnetjs.Vol(1, 1, 2)
    var loss = 0

    for (var i = 0; i < trainer.batch_size; ++i) {
        for (var j = 0; j < 100; ++j) {
            var x = convnetjs.randi(0, size)
            var y = convnetjs.randi(0, size)
            vol.w[0] = x / size - 0.5
            vol.w[1] = y / size - 0.5

            var ip = 4 * (x + size * y)
            var r = []
            r.push(p.data[ip] / 255)
            r.push(p.data[ip + 1] / 255)
            r.push(p.data[ip + 2] / 255)

            var res = trainer.train(vol, r)
            loss += res.loss
        }
    }

    console.log('Loss:', loss / trainer.batch_size / 100)
}

init()

function loop() {
    learn()
    setTimeout(loop, 1)
}

document.getElementById('start').addEventListener('click', loop)

document.getElementById('paint').addEventListener('click', function (event) {
    console.log('Painting')
    paint()
    console.log('Done painting')
})
