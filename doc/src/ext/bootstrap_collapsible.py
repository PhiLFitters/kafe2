from docutils import nodes
from docutils.parsers.rst import directives, Directive

from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective


class BootstrapCollapsibleControlNode(nodes.Element): pass

def visit_bootstrap_control_html(self, node):
    _is_button = False
    if node['control_type'] == 'button':
        _is_button = True

    # confgure control element depending on type
    if _is_button:
        _html_tag_name = 'button'
        _classes = 'btn btn-{} btn-{}'.format(node['button_style'], node['button_size'])
        _properties = {
            'data-target': '#{}'.format(node['target_id']),
            'type': 'button',
        }
    else:
        _html_tag_name = 'a'
        _classes = ''
        _properties = {
            'href': '#{}'.format(node['target_id']),
            'role': 'button',
        }

    # wrap button in 'span' tag with 'data-toggle' = 'button'
    if _is_button:
        self.body.append(self.starttag(node, 'span', **{'data-toggle': 'button'}))

    # create control element
    self.body.append(
        self.starttag(
            node,
            _html_tag_name,
            CLASS='{} collapse-control'.format(_classes),
            **dict(_properties, **{
                'data-toggle': 'collapse',
                'aria-expanded': 'false',
                'aria-controls': '#{}'.format(node['target_id']),
            })
        )
    )

    # add button/link test and close HTML tags
    self.body.append(self.starttag(node, 'span'))
    self.body.append(node['control_text'])
    self.body.append('</span>\n')
    self.body.append('</{}>\n'.format(_html_tag_name))
    if _is_button:
        self.body.append('</span>\n')


def depart_bootstrap_control(self, node):
    pass

class BootstrapCollapsibleContainerNode(nodes.Element): pass

def visit_bootstrap_container_html(self, node):
    self.body.append(
        self.starttag(
            node,
            'div',
            CLASS='bootstrap-collapsible-container collapse',
            **{
                'aria-expanded': 'False',
            }
        )
    )

def depart_bootstrap_container_html(self, node):
    self.body.append('</div>\n')

def visit_bootstrap_container_latex(self, node):
    self.visit_container(node)
def depart_bootstrap_container_latex(self, node):
    self.depart_container(node)


class BootstrapCollapsibleDirective(SphinxDirective):

    #optional_arguments = 2
    final_argument_whitespace = False
    option_spec = {
        'button_style': directives.unchanged,
        'button_size': lambda arg: directives.choice(arg, ('lg', 'sm', 'regular')),
        'control_text': directives.unchanged,
        'control_type': lambda arg: directives.choice(arg, ('button', 'link')),
    }
    has_content = True

    def run(self):
        #print(self)
        #for _k in dir(self):
        #    print("  {}: {}".format(_k, getattr(self, _k)))


        _target_id = 'collapse-{}'.format(self.env.new_serialno('collapse'))
        _target_node = nodes.target('', '', ids=[_target_id])

        _control_node = BootstrapCollapsibleControlNode(
            target_id=_target_id,
            button_style=self.options.get('button_style', 'info'),
            button_size=self.options.get('button_size', 'sm'),
            control_text=self.options.get('control_text', 'Show'),
            control_type=self.options.get('control_type', 'button'),
        )

        _content_node = BootstrapCollapsibleContainerNode()
        self.state.nested_parse(self.content, self.content_offset, _content_node)

        return [_control_node, _target_node, _content_node]


def setup(app):

    app.add_css_file('bootstrap_collapsible.css')

    app.add_node(
        BootstrapCollapsibleControlNode,
        html=(visit_bootstrap_control_html, lambda *args: None),
        latex=(lambda *args: None, lambda *args: None),
    )
    app.add_node(
        BootstrapCollapsibleContainerNode,
        html=(visit_bootstrap_container_html, depart_bootstrap_container_html),
        latex=(visit_bootstrap_container_latex, depart_bootstrap_container_latex),
    )
    app.add_directive('bootstrap_collapsible', BootstrapCollapsibleDirective)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
