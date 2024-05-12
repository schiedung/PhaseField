# MOB

This is a small program for texting new mobility rescaling

## Theory

### Helmholtz Energy Functional
$$F[\phi] = \int_\Omega f(\phi, \nabla \phi) d\Omega$$

$$\frac{\delta F}{\delta \phi} = -\frac{\pi^2}{8\eta}M_{\phi}\left(\frac{\partial f}{\partial \phi} - \nabla \cdot \frac{\partial f}{\partial \nabla \phi}\right)$$

$$f(\phi, \nabla \phi) = \sigma \left(-\frac{4\eta}{\pi^2}\nabla \phi \cdot \nabla \phi + \frac{4}{\eta} \left\vert \phi\left(1 - \phi\right)\right\vert\right)$$

### Normalization

$$ \phi      = \frac{1}{2} -\frac{1}{2}\sin\left(\frac{\pi}{\eta}\left(x-vt\right)\right) $$
$$ \phi^'    =-\frac{\pi}{2\eta}\cos\left(\frac{\pi}{\eta}\left(x-vt\right)\right) $$
$$ \phi^{''} = \frac{\pi^2}{2\eta^2}\sin\left(\frac{\pi}{\eta}\left(x-vt\right)\right) $$

$$ \phi^'    = -\frac{\pi}{\eta}\sqrt{\phi(1-\phi)} $$
$$ \phi^{''} =  \frac{\pi^2}{\eta^2}(\frac{1}{2}-\phi) $$

$$ f(\phi) = \frac{8}{\eta}\phi(1-\phi) $$
$$ F\left[\phi\right] = \sigma \int_{-\eta/2}^{\eta/2} \frac{8}{\eta}\phi(1-\phi) dx = \sigma $$

### Temporal evolution
$$\frac{\partial f}{\partial \phi} = -\sigma \frac{4}{\eta} \left(1 - 2\phi\right)$$
$$\frac{\partial f}{\partial \nabla \phi} = \sigma \frac{8\eta}{\pi^2} \Delta \phi$$
$$\dot{\phi} = M_{\phi}\sigma\left( \Delta \phi - \frac{\pi^2}{2\eta^2} \left(\frac{1}{2} - \phi\right)\right)$$
