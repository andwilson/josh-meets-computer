import React from 'react';
import Link from 'gatsby-link';
import styled from 'styled-components';

import "../styles/normalize.css";
import "../styles/prismjs.css";

const Wrapper = styled.div`
  max-width: 960px;
  margin: auto;
`;

class Template extends React.Component {
  render() {
    const { location, children } = this.props
    let header
    if (location.pathname === '/') {
      header = (
        <div></div>
      )
    } else {
      header = (
          <h3>
            <Link to={'/'}>
              Josh Meets Computer
            </Link>
          </h3>
      )
    }
    return (
      <Wrapper>
        {header}
        {children()}
      </Wrapper>
    )
  }
}

Template.propTypes = {
  children: React.PropTypes.func,
  location: React.PropTypes.object,
  route: React.PropTypes.object,
}

export default Template
